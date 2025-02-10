"""
Code from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
"""
import numpy as np
import torch
import torch.nn as nn


import torch.fft


def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))


def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    if norm is None:
              N-1
    y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
              n=0

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def code_test_dct3d():
    # init a tensor of shape [100, 20, 3]
    x = torch.rand(100, 20, 3)

    dct_coef = dct_3d(x, norm="ortho")
    print("inp signal shape: ", x.shape, "  dct coef shape: ", dct_coef.shape)

    x_recon = idct_3d(dct_coef, norm="ortho")
    print("inp signal shape: ", x.shape, "  recon signal shape: ", x_recon.shape)

    print("max error: ", torch.max(torch.abs(x - x_recon)))

    dct_coef[:, 0, :] = 0

    x_recon = idct_3d(dct_coef, norm="ortho")
    print("max error after removing first order: ", torch.max(torch.abs(x - x_recon)))


def unwarp_phase(phase, frequency_array):
    phase_lambda = torch.pi / frequency_array

    phase = phase + phase_lambda

    num_unwarp = phase // (2.0 * phase_lambda)
    phase = phase - num_unwarp * phase_lambda * 2.0

    phase = phase - phase_lambda

    return phase


def get_mag_phase(fft_weights, s=3.0 / 16.0):
    """
    Args:
        fft_weights: [*bs, numK * 2, 3/2] # [B**, numK, 2]
    Returns:
        mag_phase: [*bs, numK * 2, 3/2]
    """

    num_K = fft_weights.shape[-2] // 2

    # [num_k, 1]
    k_list = torch.arange(1, num_K + 1, device=fft_weights.device).unsqueeze(-1)
    # k_list = torch.ones_like(k_list) # need to fix this
    k_list = torch.pi * 2 * k_list * s

    _t_shape = fft_weights.shape[:-2] + (num_K, 1)
    k_list.expand(_t_shape)  # [B**, numK, 1]

    # [*bs, numK, 3/2]
    a, b = torch.split(fft_weights, num_K, dim=-2)

    # [B**, numK, 3/2]
    mag = torch.sqrt(a**2 + b**2 + 1e-10)

    sin_k_theta = -1.0 * b / (mag.detach())  # Do I need to detach?
    cos_k_theta = a / (mag.detach())  # Do I need to detach here?

    # [-pi, pi]
    k_theta = torch.atan2(sin_k_theta, cos_k_theta)
    theta = k_theta / k_list

    # [B**, numK * 2, 3/2]
    mag_phase = torch.cat([mag, theta], dim=-2)

    return mag_phase


def get_fft_from_mag_phase(mag_phase, s=3.0 / 16.0):
    """
    Args:
        mag_phase: [*bs, numK * 2, 3/2] # [B**, numK, 2]
    Returns:
        fft_weights: [*bs, numK * 2, 3/2]
    """

    num_K = mag_phase.shape[-2] // 2

    k_list = torch.arange(1, num_K + 1, device=mag_phase.device).unsqueeze(-1)
    # k_list = torch.ones_like(k_list) # need to fix this
    k_list = torch.pi * 2 * k_list * s  # scale to get frequency

    _t_shape = mag_phase.shape[:-2] + (num_K, 1)
    k_list.expand(_t_shape)  # [B**, numK, 1]

    # [*bs, numK, 3/2]
    mag, phase = torch.split(mag_phase, num_K, dim=-2)

    theta = phase * k_list

    a = mag * torch.cos(theta)
    b = -1.0 * mag * torch.sin(theta)

    # [B**, numK * 2, 3/2]
    fft_weights = torch.cat([a, b], dim=-2)

    return fft_weights


def get_displacements_from_fft_coeffs(fft_coe, t, s=3.0 / 16.0):
    """
    Args:
        fft_coe: [*bs, numK * 2, 3/2]
        t: [*bs, 1]

    Returns:
        disp = a * cos(freq * t) - b * sin(freq * t).
            Note that some formulation use
            disp = a * cos(freq * t) + b * sin(freq * t)
        shape of disp: [*bs, 3/2]
    """
    num_K = fft_coe.shape[-2] // 2
    k_list = torch.arange(1, num_K + 1, device=fft_coe.device)
    # [num_K, 1]
    freq_array = (torch.pi * 2 * k_list * s).unsqueeze(-1)

    # expand front dims to match t
    _tmp_shape = t.shape[:-1] + freq_array.shape
    freq_array.expand(_tmp_shape)  # [*bs, num_K, 1]

    cos_ = torch.cos(freq_array * t.unsqueeze(-2))
    sin_ = -1.0 * torch.sin(freq_array * t.unsqueeze(-2))

    # [*bs, num_K * 2] => [*bs, num_K]
    basis = torch.cat([cos_, sin_], dim=-2).squeeze(dim=-1)  #

    # [*bs, num_K * 2, 3/2] => [*bs, 3/2]
    disp = (basis.unsqueeze(-1) * fft_coe).sum(dim=-2)

    return disp


def bandpass_filter(signal: torch.Tensor, low_cutoff, high_cutoff, fs: int):
    """
    Args:
        signal: [T, ...]
        low_cutoff: float
        high_cutoff: float
        fs: int
    """
    # Apply FFT
    fft_signal = torch.fft.fft(signal, dim=0)
    freq = torch.fft.fftfreq(signal.size(0), d=1 / fs)

    # Bandpass filter
    mask = (freq <= low_cutoff) | (freq >= high_cutoff)
    fft_signal[mask] = 0

    # Apply inverse FFT
    filtered_signal = torch.fft.ifft(fft_signal, dim=0)
    return filtered_signal.real


def bandpass_filter_numpy(signal: np.ndarray, low_cutoff, high_cutoff, fs):
    # Apply FFT
    fft_signal = np.fft.fft(signal, axis=0)
    freq = np.fft.fftfreq(signal.shape[0], d=1 / fs)

    # Bandpass filter
    fft_signal[(freq <= low_cutoff) | (freq >= high_cutoff)] = 0

    # Apply inverse FFT
    filtered_signal = np.fft.ifft(fft_signal, axis=0)
    return filtered_signal.real


if __name__ == "__main__":
    code_test_dct3d()
