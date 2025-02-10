"""
Modified from https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py
"""
import math
from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np
from omegaconf import ListConfig
from sgm.util import append_dims, instantiate_from_config
from sgm.modules.encoders.modules import GeneralConditioner
import random


class SVDConditioner(GeneralConditioner):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__(emb_models)
        
    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        
        if force_zero_embeddings is None:
            force_zero_embeddings = []
        
            if self.training:
                img_ucg_rate = 0
                for embedder in self.embedders:
                    if embedder.input_key == "cond_frames_without_noise":
                        img_ucg_rate = embedder.ucg_rate 
                        break
                if img_ucg_rate > 0:
                    if random.random() < img_ucg_rate:
                        force_zero_embeddings.append("cond_frames_without_noise")
                        force_zero_embeddings.append("cond_frames")

        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                
                if (
                    hasattr(embedder, "input_key")
                    and embedder.input_key in force_zero_embeddings
                ):
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb
        return output
        