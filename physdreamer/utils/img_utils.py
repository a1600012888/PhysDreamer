import torch
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def make_grid(imgs: torch.Tensor, scale=0.5):
    """
    Args:
        imgs: [B, C, H, W] in [0, 1]
    Output:
        x row of images, and 3 x column of images
        which means 3 x ^ 2 <= B

        img_grid: np.ndarray, [H', W', C]
    """

    B, C, H, W = imgs.shape

    num_row = int(np.sqrt(B / 3))
    if num_row < 1:
        num_row = 1
    num_col = int(np.ceil(B / num_row))

    img_grid = torchvision.utils.make_grid(imgs, nrow=num_col, padding=0)

    img_grid = img_grid.permute(1, 2, 0).cpu().numpy()

    # resize by scale
    img_grid = cv2.resize(img_grid, None, fx=scale, fy=scale)
    return img_grid


def compute_psnr(img1, img2, mask=None):
    """
    Args:
        img1: [B, C, H, W]
        img2: [B, C, H, W]
        mask: [B, 1, H, W] or [1, 1, H, W] or None
    Outs:
        psnr: [B]
    """
    # batch dim is preserved
    if mask is None:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        if mask.shape[0] != img1.shape[0]:
            mask = mask.repeat(img1.shape[0], 1, 1, 1)
        if mask.shape[1] != img1.shape[1]:
            mask = mask.repeat(1, img1.shape[1], 1, 1)

        diff = ((img1 - img2)) ** 2
        diff = diff * mask
        mse = diff.view(img1.shape[0], -1).sum(1, keepdim=True) / (
            mask.view(img1.shape[0], -1).sum(1, keepdim=True) + 1e-8
        )

    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def torch_rgb_to_gray(image):
    # image is [B, C, H, W]
    gray_image = (
        0.299 * image[:, 0, :, :]
        + 0.587 * image[:, 1, :, :]
        + 0.114 * image[:, 2, :, :]
    )
    gray_image = gray_image.unsqueeze(1)

    return gray_image


def compute_gradient_loss(pred, gt, mask=None):
    """
    Args:
        pred: [B, C, H, W]
        gt: [B, C, H, W]
        mask: [B, 1, H, W] or None
    """
    assert pred.shape == gt.shape, "a and b must have the same shape"

    pred = torch_rgb_to_gray(pred)
    gt = torch_rgb_to_gray(gt)

    sobel_kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device
    )
    sobel_kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device
    )

    gradient_a_x = (
        torch.nn.functional.conv2d(
            pred.repeat(1, 3, 1, 1),
            sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            padding=1,
        )
        / 3
    )
    gradient_a_y = (
        torch.nn.functional.conv2d(
            pred.repeat(1, 3, 1, 1),
            sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            padding=1,
        )
        / 3
    )
    # gradient_a_magnitude = torch.sqrt(gradient_a_x ** 2 + gradient_a_y ** 2)

    gradient_b_x = (
        torch.nn.functional.conv2d(
            gt.repeat(1, 3, 1, 1),
            sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            padding=1,
        )
        / 3
    )
    gradient_b_y = (
        torch.nn.functional.conv2d(
            gt.repeat(1, 3, 1, 1),
            sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),
            padding=1,
        )
        / 3
    )
    # gradient_b_magnitude = torch.sqrt(gradient_b_x ** 2 + gradient_b_y ** 2)

    pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
    gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

    if mask is None:
        gradient_difference = torch.abs(pred_grad - gt_grad).mean()
    else:
        gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1, keepdim=True)[
            mask
        ].sum() / (mask.sum() + 1e-8)

    return gradient_difference


def mark_image_with_red_squares(img):
    # img, torch.Tensor of shape [B, H, W, C]

    mark_color = torch.tensor([1.0, 0, 0], dtype=torch.float32)

    for x_offset in range(4):
        for y_offset in range(4):
            img[:, x_offset::16, y_offset::16, :] = mark_color

    return img


# below for compute batched SSIM
def gaussian(window_size, sigma):

    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def compute_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# above for compute batched SSIM


def compute_low_res_psnr(img1, img2, scale_factor):
    """
    Args:
        img1: [B, C, H, W]
        img2: [B, C, H, W]
        scale_factor: int
    """
    img1 = F.interpolate(
        img1, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False
    )
    img2 = F.interpolate(
        img2, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False
    )
    return compute_psnr(img1, img2)


def compute_low_res_mse(img1, img2, scale_factor):
    """
    Args:
        img1: [B, C, H, W]
        img2: [B, C, H, W]
        scale_factor: int
    """
    img1 = F.interpolate(
        img1, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False
    )
    img2 = F.interpolate(
        img2, scale_factor=1 / scale_factor, mode="bilinear", align_corners=False
    )
    loss_mse = F.mse_loss(img1, img2, reduction="mean")
    return loss_mse
