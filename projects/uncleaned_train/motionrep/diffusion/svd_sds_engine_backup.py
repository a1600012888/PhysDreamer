import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors

from sgm.modules import UNCONDITIONAL_CONFIG

from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    append_dims,
)

from motionrep.utils.svd_helpper import (
    get_batch,
    get_unique_embedder_keys_from_conditioner,
)
from einops import rearrange, repeat

import torch.nn.functional as F


class SVDSDSEngine(pl.LightningModule):
    """
    stable video diffusion engine
    """

    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str] = None,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: str = "jpg",
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
    ):
        super().__init__()
        self.input_key = input_key

        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )
        self.model.eval()

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )

        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        print("init svd engine from", path)
        if path.endswith("ckpt"):
            sd = torch.load(path, map_location="cpu")["state_dict"]
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)
        elif path.endswith("bin"):
            sd = torch.load(path, map_location="cpu")
        else:
            raise NotImplementedError

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            # print(f"Unexpected Keys: {unexpected}")
            pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

        del self.first_stage_model.decoder
        self.first_stage_model.decoder = None

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, batch, training=True):
        assert training, "DiffusionEngine forward function is only for training."

        x = self.get_input(batch)  # [BT, 3, H, W]
        T = batch["num_video_frames"]
        batch_size = x.shape[0] // T
        z = self.encode_first_stage(x)  # [BT, C, H_latent, W_latent]
        batch["global_step"] = self.global_step

        sds_grad = self.emd_sds(z, batch)
        target = (z - sds_grad).detach()

        loss_sds = 0.5 * F.mse_loss(z, target, reduction="mean") / batch_size
        log_loss_dict = {
            "loss_sds_video": loss_sds.item(),
            "grad_norm": sds_grad.norm().item(),
        }

        return loss_sds, log_loss_dict

    def forward_with_encoder_chunk(self, batch, chunk_size=2):
        with torch.no_grad():
            x = self.get_input(batch)  # [BT, 3, H, W]
            T = batch["num_video_frames"]
            batch_size = x.shape[0] // T
            z = self.encode_first_stage(x)  # [BT, C, H_latent, W_latent]
            batch["global_step"] = self.global_step
            sds_grad = self.emd_sds(z, batch)

        num_chunks = math.ceil(z.shape[0] / chunk_size)

        for n in range(num_chunks):
            end_ind = min((n + 1) * chunk_size, z.shape[0])
            x_chunk = x[n * chunk_size : end_ind]
            z_chunk_recompute = self.encode_first_stage(x_chunk)

            target_chunk = (
                z_chunk_recompute - sds_grad[n * chunk_size : end_ind]
            ).detach()

            this_chunk_size = x_chunk.shape[0]
            loss_sds_chunk = (
                0.5
                * F.mse_loss(z_chunk_recompute, target_chunk, reduction="mean")
                * this_chunk_size
                / z.shape[0]
                / batch_size
            )

            loss_sds_chunk.backward()

        with torch.no_grad():
            target = (z - sds_grad).detach()
            loss_sds = 0.5 * F.mse_loss(z, target, reduction="mean") / batch_size
            log_loss_dict = {
                "loss_sds_video": loss_sds.item(),
                "grad_norm": sds_grad.norm().item(),
            }

            video_space_sds_grad = x.grad

        return video_space_sds_grad, loss_sds, log_loss_dict

    @torch.no_grad()
    def emd_sds(self, input_x, extra_input):
        """
        Args:
            input_x: [BT, C, H, W] in latent
            extra_input: dict
                "fps_id": [B]
                "motion_bucket_id": [B]
                "cond_aug": [B]
                "cond_frames_without_noise": [B, C, H, W]
                "cond_frames": [B, C, H, W]
        """

        num_frames = extra_input["num_video_frames"]
        batch_size = input_x.shape[0] // num_frames
        device = input_x.device
        # video = video.contiguous()

        extra_input["num_video_frames"] = num_frames

        # prepare c and uc

        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.conditioner),
            extra_input,
            [1, num_frames],
            T=num_frames,
            device=device,
        )

        # keys would be be ['crossattn', 'vector', 'concat']
        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
            c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        # after this should be
        # crossattn torch.Size([14, 1, 1024])
        # vector torch.Size([14, 768])
        # concat torch.Size([14, 4, 72, 128])

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(
            int(2 * batch_size), num_frames
        ).to(device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

        sigmas = self.loss_fn.sigma_sampler(batch_size).to(input_x)
        sigmas = repeat(sigmas, "b ... -> b t ...", t=num_frames)
        sigmas = rearrange(sigmas, "b t ... -> (b t) ...", t=num_frames)

        noise = torch.randn_like(input_x)  # [BT, C, H, W]

        sigmas_bc = append_dims(sigmas, input_x.ndim)  # [14, 1, 1, 1]
        noised_input = self.loss_fn.get_noised_input(
            sigmas_bc, noise, input_x
        )  # [BT, C, H, W]

        # [2BT, C, H, W], [2BT]
        bathced_xt, bathced_sigmas, bathched_c = self.sampler.guider.prepare_inputs(
            noised_input, sigmas, c, uc
        )
        # bathched_c[crossattn] => [2BT, 1, C] ;   bathched_c["concat"] => [2BT, C, H, W]; bathched_c["vector"] => [2BT, C_feat]

        # output shape [2BT, C, H, W]

        denoised = self.denoiser(
            self.model,
            bathced_xt,
            bathced_sigmas,
            bathched_c,
            **additional_model_inputs,
        )

        # [BT, C, H, W]
        denoised = self.sampler.guider(denoised, bathced_sigmas)

        sds_grad = (denoised - input_x) / sigmas_bc

        return sds_grad
