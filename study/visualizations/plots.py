import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from keras.utils import plot_model

from study.models.layers import get_augmentation_layer
from study.models import build_model
from lib.legacy.svd import compress_mc, compress_sc

from util.util import (
    normalize_img,
    # absolute_err,
    elementwise_err,
    # mean_absolute_err,
    # total_absolute_err,
)
from util.constants import PATHS, HRSTC, DFLTS
from util.trial_detail import TrialDetail

from compression_layer import RandomCompression


plt.style.use("ggplot")
mpl.rcParams["font.family"] = "STIXGeneral"


def compare_compressed(img, name):
    compressed = compress_mc(img).compressed
    complement = elementwise_err(img, compressed)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    fig.suptitle("Original vs Compressed", fontsize=12)

    axs[0].imshow(img)
    axs[0].set_title("Original")

    compressed = normalize_img(compressed)
    axs[1].imshow(compressed)
    axs[1].set_title("Compressed")

    complement = normalize_img(complement)
    axs[2].imshow(complement)
    axs[2].set_title("Error")

    for ax in axs:
        ax.axis("off")

    _save_fig(name, "comparison_plot")
    plt.show()


def scree(img, name, energy_factor=HRSTC["MAX_ENERGY_FACTOR"]):
    compressed, R, G, B, *_ = compress_mc(img, energy_factor)
    rs = R.S
    gs = G.S
    bs = B.S
    rs = rs[:50]
    gs = gs[:50]
    bs = bs[:50]

    """
    Calculate cumulative sum along the last axis
    and normalize by the sum along the last axis
    """
    s = np.stack((rs, gs, bs), axis=-1)
    props_rgb = np.apply_along_axis(lambda x: np.cumsum(x) / np.sum(x), axis=-1, arr=s)

    # Calculate mean along the first axis
    props_rgb_mean = np.mean(props_rgb, axis=0)
    k = np.argmin(np.abs(props_rgb_mean - energy_factor)) + 1
    print(k)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(
        "Scree Plot of the Singular Values Profiles of Different Channels", fontsize=12
    )

    axs[0].imshow(compressed)
    axs[0].axis("off")
    axs[0].set_title("Image", fontsize=10)

    axs[1].plot(
        np.arange(1, len(rs) + 1),
        rs,
        marker=".",
        color="r",
        alpha=0.3,
        label="Red Channel",
    )
    axs[1].plot(
        np.arange(1, len(gs) + 1),
        gs,
        marker="v",
        color="g",
        alpha=0.3,
        label="Green Channel",
    )
    axs[1].plot(
        np.arange(1, len(bs) + 1),
        bs,
        marker="*",
        color="b",
        alpha=0.3,
        label="Red Channel",
    )
    axs[1].grid(True, linestyle="-", alpha=0.7)
    axs[1].axvline(x=k, color="k", linestyle="--", alpha=0.7)  # Vertical line
    axs[1].legend()
    axs[1].set_title("Image Singular Values of Different Channels", fontsize=10)
    axs[1].set_xlabel("Truncated Indeces of Singular Values")
    axs[1].set_ylabel("Singular Values")

    _save_fig(name, "scree_plot")
    plt.show()


def indiv_chan(img, name, energy_factor=HRSTC["MAX_ENERGY_FACTOR"]):
    compressed, R, G, B, *_ = compress_mc(img, energy_factor)
    R = R.compressed
    G = G.compressed
    B = B.compressed

    num_rows = 2
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Channel-Wise SVD", fontsize=12)

    titles = ["R", "G", "B", "RGB"]

    images = [
        (img[..., 0], "Reds"),
        (img[..., 1], "Greens"),
        (img[..., 2], "Blues"),
        (img, None),
    ]
    for i, (image, cmap) in enumerate(images):
        axs[0, i].imshow(image, cmap=cmap)
        axs[0, i].set_title(titles[i])

    images = [(R, "Reds"), (G, "Greens"), (B, "Blues"), (compressed, None)]
    for i, (image, cmap) in enumerate(images):
        axs[1, i].imshow(image, cmap=cmap)
        axs[1, i].set_title(titles[i])

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "indiv_chan")
    plt.show()


def sc_indiv_r1_mats(img, name):
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)

    compressed_result = compress_sc(img, 0)

    num_rows = 2
    num_cols = 5
    num_plots = num_rows * num_cols

    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Rank 1 Matrices of Decreasing Singular Values", fontsize=12)

    axs[0, 0].imshow(img, cmap="gray")
    axs[0, 0].set_title("Original Image", fontsize=9)

    for i in range(1, num_plots):
        row, col = divmod(i, num_cols)

        r1_img = _get_cum_r1_mat(compressed_result, upper_bd=i, lower_bd=i - 1)
        normalized_img = normalize_img(r1_img)

        axs[row, col].imshow(normalized_img, cmap="gray")
        axs[row, col].set_title(f"R1 Matrix {i}", fontsize=9)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "sc_indiv_r1_mats")
    plt.show()


def mc_indiv_r1_mats(img, name):
    _, R, G, B, *_ = compress_mc(img, 0)

    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols)

    axs[0, 0].imshow(img, cmap="gray")
    fig.suptitle("Rank 1 Matrices of Decreasing Singular Values", fontsize=12)

    for i in range(1, 10):
        row, col = divmod(i, 5)

        red = _get_cum_r1_mat(R, upper_bd=i, lower_bd=i - 1)
        green = _get_cum_r1_mat(G, upper_bd=i, lower_bd=i - 1)
        blue = _get_cum_r1_mat(B, upper_bd=i, lower_bd=i - 1)

        r1_img = np.stack([red, green, blue], axis=-1)

        normalized_img = normalize_img(r1_img)

        axs[row, col].imshow(normalized_img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "mc_indiv_r1_mats")
    plt.show()


def sc_cum_r1_mats(img, name):
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)

    cum_apprx = np.zeros(shape=img.shape)
    compressed_result = compress_sc(img, 0)

    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Cumulative Low Rank Approximations", fontsize=12)

    for i in range(10):
        row, col = divmod(i, 5)
        upper_bd = (i + 1) * 3

        r1_img = _get_cum_r1_mat(compressed_result, upper_bd)

        cum_apprx = np.add(cum_apprx, r1_img)

        axs[row, col].imshow(cum_apprx, cmap="gray")

    axs[1, 4].imshow(img, cmap="gray")

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "sc_cum_r1_mats")
    plt.show()


def mc_cum_r1_mats(img, name):
    cum_apprx = np.zeros(shape=img.shape)

    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Cumulative Low Rank Approximations", fontsize=12)

    _, R, G, B, *_ = compress_mc(img, 0)

    for i in range(10):
        row, col = divmod(i, 5)
        upper_bd = (i + 1) * 3

        red = _get_cum_r1_mat(R, upper_bd)
        green = _get_cum_r1_mat(G, upper_bd)
        blue = _get_cum_r1_mat(B, upper_bd)

        ith_img = np.stack([red, green, blue], axis=-1)

        cum_apprx = np.add(cum_apprx, ith_img)

        normalized_img = normalize_img(cum_apprx)

        axs[row, col].imshow(normalized_img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "mc_cum_r1_mats")
    plt.show()


def img_manifold_hypothesis(name):
    num_rows = 4
    num_cols = 4
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.suptitle("Image Manifold Hypothesis", fontsize=12)
    plt.tight_layout()

    shape = (180, 180)

    for i in range(4 * 4):
        row, col = divmod(i, 4)

        unif_noise_R = np.random.uniform(low=0, high=1, size=shape)
        unif_noise_G = np.random.uniform(low=0, high=1, size=shape)
        unif_noise_B = np.random.uniform(low=0, high=1, size=shape)

        ith_img = np.stack([unif_noise_R, unif_noise_G, unif_noise_B], axis=-1)

        axs[row, col].imshow(ith_img)

    _save_fig(name, "img_manifold_hypothesis")

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    plt.show()


def sc_complement_mat(img, name):
    if len(img.shape) == 3:
        img = np.mean(img, axis=-1)

    compressed = compress_sc(img, 0)

    factor = 5

    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    fig.suptitle("Low Rank Approximations and their Complements", fontsize=12)

    for i in range(9):
        boundary = (i + 1) * factor

        img_apprx = _get_cum_r1_mat(compressed, upper_bd=boundary)
        img_complement = elementwise_err(img, img_apprx)

        axs[0, i % num_cols].imshow(img_apprx, "gray")
        axs[1, i % num_cols].imshow(img_complement, "gray")

    axs[0, 4].imshow(img, "gray")
    axs[1, 4].imshow(np.ones(shape=img.shape), "gray")

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "sc_complement")
    plt.show()


def mc_complement_mat(img, name):
    _, R, G, B, _, rank, *_ = compress_mc(img, 0)

    num_rows = 2
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    fig.suptitle("Low Rank Approximations and their Complements", fontsize=12)

    factor = 5

    for i in range(9):
        boundary = (i + 1) * factor

        apprx_R = _get_cum_r1_mat(R, upper_bd=boundary)
        apprx_G = _get_cum_r1_mat(G, upper_bd=boundary)
        apprx_B = _get_cum_r1_mat(B, upper_bd=boundary)

        apprx = np.stack([apprx_R, apprx_G, apprx_B], axis=-1)

        normalized_apprx = normalize_img(apprx)

        axs[0, i % num_cols].imshow(normalized_apprx)

        complement_R = elementwise_err(R.compressed, img[..., 0])
        complement_G = elementwise_err(G.compressed, img[..., 1])
        complement_B = elementwise_err(B.compressed, img[..., 2])

        complement = np.stack([complement_R, complement_G, complement_B], axis=-1)

        normalized_complement = normalize_img(complement)

        axs[1, i % num_cols].imshow(normalized_complement)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    axs[0, 4].imshow(img)
    axs[1, 4].imshow(np.zeros(shape=img.shape))

    _save_fig(name, "mc_adjoint")
    plt.show()


def plot_all_augmentations(img, name):
    data_augmentation = get_augmentation_layer("all", energy_factor=0.50)
    batched_rgb_image = np.expand_dims(img, axis=0)

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Default Augmentations + SVD", fontsize=12)

    for i in range(1, 9):
        row, col = divmod(i, 3)
        augmented = data_augmentation(batched_rgb_image)[0]
        normalized_img = normalize_img(augmented)

        axs[row, col].imshow(normalized_img)

    img = normalize_img(img)
    axs[0, 0].imshow(img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "augmentation_variations")
    plt.show()


def constant_distortions(img, name):
    data_augmentation = RandomCompression(
        distribution="constant", max_energy_factor=0.5
    )
    batched_rgb_image = np.expand_dims(img, axis=0)

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Constant Low Rank Approximations", fontsize=12)

    for i in range(1, 9):
        row, col = divmod(i, 3)
        augmented = data_augmentation(batched_rgb_image)[0]
        normalized_img = normalize_img(augmented)

        axs[row, col].imshow(normalized_img)

    img = normalize_img(img)
    axs[0, 0].imshow(img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "deterministic_constant_distortions")
    plt.show()


def random_uniform_distortions(img, name):
    data_augmentation = RandomCompression(
        distribution="uniform",
        min_energy_factor=0.2,
        max_energy_factor=0.6,
    )
    batched_rgb_image = np.expand_dims(img, axis=0)

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Random Uniform Low Rank Approximations", fontsize=12)

    for i in range(1, 9):
        row, col = divmod(i, 3)
        augmented = data_augmentation(batched_rgb_image)[0]
        normalized_img = normalize_img(augmented)

        axs[row, col].imshow(normalized_img)

    img = normalize_img(img)
    axs[0, 0].imshow(img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "random_uniform_distortions")
    plt.show()


def random_gaussian_distortions(img, name):
    data_augmentation = RandomCompression(distribution="normal", max_energy_factor=0.4)
    batched_rgb_image = np.expand_dims(img, axis=0)

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols)
    fig.suptitle("Random Gaussian Low Rank Approximations", fontsize=12)

    for i in range(1, 9):
        row, col = divmod(i, 3)
        augmented = data_augmentation(batched_rgb_image)[0]
        normalized_img = normalize_img(augmented)

        axs[row, col].imshow(normalized_img)

    img = normalize_img(img)
    axs[0, 0].imshow(img)

    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    _save_fig(name, "random_gaussian_distortions")
    plt.show()


def model_architectures():
    trial_details = [
        TrialDetail(ds_name, ag_mthd)
        for ds_name in DFLTS["DATASETS"]
        for ag_mthd in DFLTS["AUGMENTATION_METHODS"]
    ]

    architectures_path = os.path.join(PATHS["RESOURCES"], "architectures")
    os.makedirs(architectures_path, exist_ok=True)

    for i, trial_detail in enumerate(trial_details):
        model = build_model(trial_detail)
        file_path = os.path.join(architectures_path, f"model_diagram_{i}.png")

        plot_model(
            model,
            to_file=file_path,
            show_shapes=True,
            show_layer_names=True,
            expand_nested=True,
            rankdir="TB",
        )


def _get_cum_r1_mat(compressed, upper_bd, lower_bd=0):
    U_trunc = compressed.U[:, lower_bd:upper_bd]
    S_trunc = np.diag(compressed.S[lower_bd:upper_bd])
    V_trunc = compressed.V[lower_bd:upper_bd, :]

    return U_trunc @ S_trunc @ V_trunc


def _save_fig(img_name, fig_name):
    plots_path = os.path.join(PATHS["PLOTS"], img_name)
    os.makedirs(plots_path, exist_ok=True)
    save_path = os.path.join(plots_path, f"{fig_name}.png")
    plt.savefig(save_path)


# def compare_spectral_vs_rank(img, name):
#     if len(img.shape) == 3:
#         img = np.mean(img, axis=-1)

#     percentiles = [.01, .05, .10, .20, .50]

#     def spectral_norm_percentiles(S):
#         total_spec_norm = np.sum(S)
#         cumulative_sum = np.cumsum(S)
#         percentiles_values = [np.percentile(cumulative_sum, p)
#                               for p in percentiles]
#         return percentiles_values

#     def rank_prop_percentiles(S):
#         percentile_values = [len(S) * perc for perc in percentiles]
#         return percentile_values

#     singular_vals = compress_sc(img).S

#     plt.figure(figsize=(8, 6))

#     # Plot singular values
#     plt.plot(np.arange(1, len(singular_vals) + 1), singular_vals, marker='.', color='r', label='Singular Values')

#     spec_percentiles = spectral_norm_percentiles(singular_vals)
#     for i, percentile in enumerate(spec_percentiles, start=1):
#         label_percentile = percentiles[i-1] * 100
#         plt.axvline(x=percentile, linestyle='--', color='b', alpha=0.7, label=f'Spectral {label_percentile:.2f}%')

#     # Plot rank proportion percentiles
#     rank_percentiles = rank_prop_percentiles(singular_vals)
#     for i, percentile in enumerate(rank_percentiles, start=1):
#         label_percentile = percentiles[i-1] * 100
#         plt.axvline(x=percentile, ='--', color='g', alpha=0.7, label=f'Rank {label_percentile:.2f}%')

#     plt.title('Spectral Norm Proportion vs Rank Proportion', fontsize=12)
#     plt.xlabel('Rank', fontsize=10)
#     plt.ylabel('Spectral Norm Proportion / Percentiles', fontsize=10)
#     plt.legend()

#     plt.savefig(name + '_spectral_vs_rank.png')
#     plt.show()


# def compare_svd_tf_sp(img, name):
#     batched_rgb_image = np.expand_dims(img, axis=0)

#     fig, axs = plt.subplots(1, 2)
#     fig.suptitle('TF SVD vs SciPy SVD', fontsize=12)

#     # TF SVD
#     tf_compression = compress_image_with_energy(img, .5)
#     # tf_compression = normalize_img(tf_compression)
#     axs[0].imshow(tf_compression)

#     # TF SVD
#     scipy_compression = compress_mc(img, .5).compressed
#     scipy_compression = normalize_img(scipy_compression)
#     axs[1].imshow(scipy_compression)

#     for ax in axs:
#       ax.axis('off')

#     _save_fig(name, 'TFSVD_vs_SciPySVD')
#     plt.show()


# def compare_rgb_adj(img, name):
#     compressed_result = compress_mc(img)
#     adjusted = adjust_rgb_corr(img, compressed_result, 3)

#     total_orig_ae = total_absolute_err(img, img)
#     total_compressed_ae = total_absolute_err(img, compressed_result.compressed)
#     total_adjusted_ae = total_absolute_err(img, adjusted)

#     fig, axs = plt.subplots(1, 3, figsize=(10, 5))
#     fig.suptitle('Original vs Compressed', fontsize=12)

#     axs[0].imshow(compressed_result.compressed)
#     axs[1].imshow(adjusted)
#     axs[2].imshow(img)

#     axs[0].set_title('Compressed')
#     axs[1].set_title('Compressed + Adjusted')
#     axs[2].set_title('Original')

#     axs[0].text(0.5, -0.1, f'AE: {total_compressed_ae}', transform=axs[0].transAxes, ha='center', fontsize=10)
#     axs[1].text(0.5, -0.1, f'AE: {total_adjusted_ae}', transform=axs[1].transAxes, ha='center', fontsize=10)
#     axs[2].text(0.5, -0.1, f'AE: {total_orig_ae}', transform=axs[2].transAxes, ha='center', fontsize=10)

#     for ax_row in axs:
#         for ax in ax_row:
#             ax.axis('off')

#     _save_fig(name, 'comp_adjusted')
#     plt.show()


# def sc_noise_vs_svd(img, name):
#     if len(img.shape) == 3:
#       img = np.mean(img, axis=-1)

#     compressed = compress_sc(img, 0).compressed

#     svd_ae = absolute_err(img, compressed)
#     svd_mae = mean_absolute_err(img, compressed)

#     # Expectation of a unif r.v. times the number of channels
#     eps = (svd_mae) // (2 * 3 * img.shape[0] * img.shape[1])
#     unif_noise = np.random.uniform(low=-eps, high=eps, size=img.shape)
#     noisy_img = img + unif_noise
#     normalized_noise = normalize_img(noisy_img)

#     num_rows = 2; num_cols = 5
#     fig, axs = plt.subplots(num_rows, num_cols)

#     axs[0, 0].imshow(svd_ae, "gray")
#     axs[0, 0].text(0.5, -0.1, f'MAE: {svd_mae:.5}', size=8, ha="center", transform=axs[0, 0].transAxes)
#     axs[0, 1].imshow(compressed, "gray")
#     axs[0, 1].text(0.5, -0.1, 'SVD Compressed', size=8, ha="center", transform=axs[0, 1].transAxes)
#     axs[1, 0].imshow(unif_noise, "gray")
#     axs[1, 0].text(0.5, -0.1, f'MAE: {np.sum(np.abs(unif_noise)):.5}', size=8, ha="center", transform=axs[1, 0].transAxes)
#     axs[1, 1].imshow(normalized_noise, "gray")
#     axs[1, 1].text(0.5, -0.1, 'Uniform Noise Injected', size=8, ha="center", transform=axs[1, 1].transAxes)

#     for ax_row in axs:
#        for ax in ax_row:
#           ax.axis('off')

#     resources_path = f'{RESOURCES_DIR}/{name}'
#     os.makedirs(resources_path, exist_ok=True)
#     save_path = os.path.join(resources_path, 'noise_vs_svd.png')
#     plt.savefig(save_path)

#     plt.show()
