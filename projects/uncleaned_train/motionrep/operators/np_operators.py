import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def feature_map_to_rgb_pca(feature_map):
    """
    Args:
        feature_map: (C, H, W) feature map.
    Outputs:
        rgb_image: (H, W, 3) image.
    """
    # Move feature map to CPU and convert to numpy
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    H, W = feature_map.shape[1:]
    # Flatten spatial dimensions  # [N, C]
    flattened_map = feature_map.reshape(feature_map.shape[0], -1).T

    # Apply PCA and reduce channel dimension to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flattened_map)

    # Reshape back to (H, W, 3)
    rgb_image = pca_result.reshape(H, W, 3)

    # Normalize to [0, 1]
    rgb_image = (rgb_image - rgb_image.min()) / (
        rgb_image.max() - rgb_image.min() + 1e-3
    )

    return rgb_image
