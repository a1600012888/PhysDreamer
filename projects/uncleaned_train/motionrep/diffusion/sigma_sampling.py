import torch
from inspect import isfunction

# import sgm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class EDMSamplingWithResShift:
    def __init__(self, p_mean=-1.2, p_std=1.2, scale_shift=320.0 / 576):
        self.p_mean = p_mean
        self.p_std = p_std
        self.scale_shift = scale_shift

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))

        sigma = log_sigma.exp() * self.scale_shift
        return sigma
