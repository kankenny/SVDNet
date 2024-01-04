import numpy as np
from PIL import Image
from scipy import linalg


ex_img_path = '/workspaces/workspace/res/cat.jpg'

img = Image.open(ex_img_path)
np_img = np.array(img)


def compress_sc(a, comp_rate=0.2):
    """
    Single-Channel image compression using Singular Value Decomposition

    I: a - matrix-like arr (2d), comp_rate - rate of rank 1 matrices disregarded
    O: compressed_img - sum of rank 1 matrices 
    """
    U, S, V = linalg.svd(a)
    
    k = int((1 - comp_rate) * min(a.shape))
    
    U_trunc = U[:, :k]
    S_trunc = np.diag(S[:k])
    V_trunc = V[:k, :]
    
    compressed_img = U_trunc @ S_trunc @ V_trunc
    
    return compressed_img

def compress_mc(a, comp_rate=0.2):
    """
    Multi-Channel image compression using Singular Value Decomposition

    I: a - matrix-like arr (3d), comp_rate - rate of rank 1 matrices disregarded
    O: compressed_img - sum of rank 1 matrices 
    """
    R = compress_sc(a[..., 0], comp_rate=comp_rate)
    G = compress_sc(a[..., 1], comp_rate=comp_rate)
    B = compress_sc(a[..., 2], comp_rate=comp_rate)

    compressed_img = np.stack([R, G, B], axis=-1)
    
    return compressed_img

compress_mc(np_img)