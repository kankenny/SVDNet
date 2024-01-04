import numpy as np
from PIL import Image
from scipy import linalg


ex_img_path = '/workspaces/workspace/res/cat.jpg'

img = Image.open(ex_img_path)
np_img = np.array(img)


def compress_sc(a, comp_rate=0.2):
    """
    Single-Channel image compression using Singular Value Decomposition (SVD)

    Args:
        a (numpy.ndarray): Input image as a 3D tensor (height x width x channels).
        comp_rate (float, optional): Rate of 'least valuable' rank-1 matrices disregarded (default is 0.2).
        chan_wise (bool, optional): If True, apply independent SVD on individual channels; otherwise, on the average of the channel-wise pixel values (default is True).

    Returns:
        numpy.ndarray: Compressed image as a 3D tensor.

    Raises:
        ValueError: If the input tensor 'a' is not a 3D tensor.
    """
    U, S, V = linalg.svd(a)
    
    k = int((1 - comp_rate) * min(a.shape))
    
    U_trunc = U[:, :k]
    S_trunc = np.diag(S[:k])
    V_trunc = V[:k, :]
    
    compressed_img = U_trunc @ S_trunc @ V_trunc
    
    return compressed_img

def compress_mc(a, comp_rate=0.2, chan_wise=True):
    """
    Multi-Channel image compression using SVD

    Args:
        a (numpy.ndarray): Input image as a 3D tensor (height x width x channels).
        comp_rate (float, optional): Rate of 'least valuable' rank-1 matrices disregarded (default is 0.2).
        chan_wise (bool, optional): If True, apply independent SVD on individual channels; otherwise, on the average of the channel-wise pixel values (default is True).

    Returns:
        numpy.ndarray: Compressed image as a 3D tensor.

    Raises:
        ValueError: If the input tensor 'a' is not a 3D tensor.
    """
    if len(a.shape) != 3:
        raise ValueError("Input tensor 'a' must be a 3D tensor with shape (height, width, channels).")

    if chan_wise == False:  # Average the corresponding pixel-wise values in the channel dimension
        avg = np.sum(a, axis=2) / 3
        a[..., 0:3] = avg[..., None]

    R = compress_sc(a[..., 0], comp_rate=comp_rate)
    G = compress_sc(a[..., 1], comp_rate=comp_rate)
    B = compress_sc(a[..., 2], comp_rate=comp_rate)

    compressed_img = np.stack([R, G, B], axis=-1)
    
    return compressed_img

compress_mc(np_img)