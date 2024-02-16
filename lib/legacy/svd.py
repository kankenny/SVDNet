import numpy as np
from scipy import linalg

from util.data_types.compression_result import SCCompressedImg, MCCompressedImg
from util.timer_deco import timer
from util.constants import HRSTC


@timer
def compress_sc(a, energy_factor=HRSTC["MAX_ENERGY_FACTOR"]):
    """
    Single-Channel image compression using Singular Value Decomposition (SVD)

    Args:
        a (numpy.ndarray): Input image as a 2D tensor (height x width).
        energy_factor (float, optional): Amount of energy
                                         (cumulative summation of singular values)
                                         of the image

    Returns:
        numpy.ndarray: Compressed image as a 2D tensor.
    """
    U, S, V = linalg.svd(a)

    k = _spectral_proportion(S, energy_factor)

    U_trunc = U[:, :k]
    S_trunc = np.diag(S[:k])
    V_trunc = V[:k, :]

    compressed_img = U_trunc @ S_trunc @ V_trunc

    return SCCompressedImg(compressed_img, U, S, V, energy_factor, k)


def compress_mc(a, energy_factor=HRSTC["MAX_ENERGY_FACTOR"]):
    """
    Multi-Channel image compression using SVD

    Args:
        a (numpy.ndarray): Input image as a 3D tensor (height x width x channels).
        energy_factor (float, optional): Amount of energy
                                         (cumulative summation of singular values)
                                         of the image

    Returns:
        numpy.ndarray: Compressed image as a 3D tensor.

    Raises:
        ValueError: If the input tensor 'a' is not a 3D tensor.
    """
    if len(a.shape) != 3:
        raise ValueError(
            "Input tensor 'a' must be a 3D tensor with shape (height, width, channels)."
        )

    R = compress_sc(a[..., 0], energy_factor=energy_factor)
    G = compress_sc(a[..., 1], energy_factor=energy_factor)
    B = compress_sc(a[..., 2], energy_factor=energy_factor)

    compressed_img = np.stack([R.compressed, G.compressed, B.compressed], axis=-1)

    return MCCompressedImg(compressed_img, R, G, B, energy_factor)


def _spectral_proportion(S, energy_factor):
    total_spec_norm = np.sum(S)
    cumulative_sum = np.cumsum(S)
    k = int(np.argmax(cumulative_sum / total_spec_norm >= (energy_factor)))
    return max(1, k)
