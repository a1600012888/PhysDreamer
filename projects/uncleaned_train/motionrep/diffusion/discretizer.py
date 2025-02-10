import torch

from sgm.modules.diffusionmodules.discretizer import Discretization


class EDMResShiftedDiscretization(Discretization):
    def __init__(
        self, sigma_min=0.002, sigma_max=80.0, rho=7.0, scale_shift=1.0
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.scale_shift = scale_shift

    def get_sigmas(self, n, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = sigmas * self.scale_shift
        return sigmas
