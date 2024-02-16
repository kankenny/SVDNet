import os
from glob import glob
from PIL import Image

from util.util import load_image

from study.plots import (
    compare_compressed,
    indiv_chan,
    scree,
    mc_indiv_r1_mats,
    sc_indiv_r1_mats,
    mc_complement_mat,
    sc_complement_mat,
    mc_cum_r1_mats,
    sc_cum_r1_mats,
    img_manifold_hypothesis,
    plot_all_augmentations,
    constant_distortions,
    random_uniform_distortions,
    random_gaussian_distortions,
)

from util.constants import PATHS


functions_with_img_arg = [
    compare_compressed,
    indiv_chan,
    scree,
    mc_indiv_r1_mats,
    sc_indiv_r1_mats,
    mc_complement_mat,
    sc_complement_mat,
    mc_cum_r1_mats,
    sc_cum_r1_mats,
    plot_all_augmentations,
    constant_distortions,
    random_uniform_distortions,
    random_gaussian_distortions,
]

functions_wo_img_arg = [img_manifold_hypothesis]

IMG_NAMES = [
    "cat",
    "tree",
    "digit",
    "text",
]


def visualize_all():
    for name in IMG_NAMES:
        img_path = os.path.join(PATHS["RESOURCES"], "test_images", f"{name}.jpg")
        img = load_image(img_path)

        for func in functions_with_img_arg:
            func(img, name)
        for func in functions_wo_img_arg:
            func(name)

        concat_imgs(name)


def concat_imgs(name):
    png_files = glob(os.path.join(PATHS["RESOURCES"], "plots", name, "*.png"))
    images = [Image.open(png_file) for png_file in png_files]

    if images:
        # Get the width of the first image and calculate total height
        width, _ = images[0].size
        total_height = sum(img.size[1] for img in images)

        # Create a blank image with the calculated dimensions
        concat_image = Image.new("RGB", (width, total_height), (255, 255, 255))

        # Paste each resized image into the blank image
        y_offset = 0
        for img in images:
            # Resize each image to match the width of the first image
            img = img.resize((width, int(img.size[1] * (width / img.size[0]))))
            concat_image.paste(img, (0, y_offset))
            y_offset += img.size[1]

        # Save the concatenated image
        dest_path = os.path.join(PATHS["RESOURCES"], f"{name}_concat.png")
        concat_image.save(dest_path)
        print(f'\nConcatenated image saved at {PATHS["RESOURCES"]}/plots/\n')
    else:
        print("No PNG images found in the specified directory.")
