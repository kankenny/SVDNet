import numpy as np

from util.util import normalize_img
from util.timer_deco import timer
from util.constants import HRSTC


@timer
def adjust_rgb_corr(orig, comp_result, patch_sz=3):
    """
    Warning: Computationally Intractable (time)
    Tries:
        - Vectorize: still intractable in time
        - tf.function: intractable in space



    Adjust pixel value of a specific channel based on the
    neighboring pixels contained in the patch
    """
    compressed, R, G, B, *_ = comp_result
    imgheight, imgwidth, _ = orig.shape

    copy = compressed.copy()

    for i in range(imgheight):
        for j in range(imgwidth):
            offset = (patch_sz - 1) // 2

            patch_locs = _get_patch_locs(imgheight, imgwidth, i, j, offset, patch_sz)

            r_patch = orig[patch_locs][:, :, 0]
            g_patch = orig[patch_locs][:, :, 1]
            b_patch = orig[patch_locs][:, :, 2]

            r_flat = r_patch.flatten()
            g_flat = g_patch.flatten()
            b_flat = b_patch.flatten()

            rg_corr = np.corrcoef(r_flat, g_flat)
            gb_corr = np.corrcoef(g_flat, b_flat)
            br_corr = np.corrcoef(b_flat, r_flat)

            r_eps = HRSTC["MAX_EPS"] * np.mean([rg_corr, br_corr])
            g_eps = HRSTC["MAX_EPS"] * np.mean([rg_corr, gb_corr])
            b_eps = HRSTC["MAX_EPS"] * np.mean([gb_corr, br_corr])

            copy[i, j, 0] += r_eps
            copy[i, j, 1] += g_eps
            copy[i, j, 2] += b_eps

    rgb_adj_normalized = normalize_img(copy)

    return rgb_adj_normalized


def _get_patch_locs(imgheight, imgwidth, i, j, offset, patch_sz):
    patch_top = np.clip(i - offset, 0, imgheight)
    patch_bottom = np.clip(i + offset + 1, 0, imgheight)
    patch_left = np.clip(j - offset, 0, imgwidth)
    patch_right = np.clip(j + offset + 1, 0, imgwidth)

    patch_locs = (slice(patch_top, patch_bottom), slice(patch_left, patch_right))

    return patch_locs


@timer
def test_adjust_rgb_corr(orig, comp_result, patch_sz=3):
    """
    Warning: Computationally Intractable (time)
    Tries:
        - Vectorize: still intractable in time
        - tf.function: intractable in space



    Adjust pixel value of a specific channel based on the
    neighboring pixels contained in the patch
    """
    compressed, R, G, B, *_ = comp_result
    imgheight, imgwidth, _ = orig.shape

    offset = (patch_sz - 1) // 2
    i, j = np.meshgrid(np.arange(imgheight), np.arange(imgwidth), indexing="ij")

    patch_top = np.clip(i - offset, 0, imgheight)
    patch_bottom = np.clip(i + offset + 1, 0, imgheight)
    patch_left = np.clip(j - offset, 0, imgwidth)
    patch_right = np.clip(j + offset + 1, 0, imgwidth)

    patch_locs_top = patch_top.flatten()
    patch_locs_bottom = patch_bottom.flatten()
    patch_locs_left = patch_left.flatten()
    patch_locs_right = patch_right.flatten()

    patch_locs = (
        patch_locs_top[:, np.newaxis],
        patch_locs_bottom[:, np.newaxis],
        patch_locs_left[np.newaxis, :],
        patch_locs_right[np.newaxis, :],
    )

    r_patch = orig[patch_locs[0], patch_locs[2], 0]
    g_patch = orig[patch_locs[0], patch_locs[2], 1]
    b_patch = orig[patch_locs[0], patch_locs[2], 2]

    r_flat = r_patch.flatten()
    g_flat = g_patch.flatten()
    b_flat = b_patch.flatten()

    rg_corr = np.corrcoef(r_flat, g_flat)
    gb_corr = np.corrcoef(g_flat, b_flat)
    br_corr = np.corrcoef(b_flat, r_flat)

    r_eps = HRSTC["MAX_EPS"] * np.mean([rg_corr, br_corr])
    g_eps = HRSTC["MAX_EPS"] * np.mean([rg_corr, gb_corr])
    b_eps = HRSTC["MAX_EPS"] * np.mean([gb_corr, br_corr])

    compressed[:, :, 0] += r_eps
    compressed[:, :, 1] += g_eps
    compressed[:, :, 2] += b_eps

    rgb_adj_normalized = normalize_img(compressed)

    return rgb_adj_normalized
