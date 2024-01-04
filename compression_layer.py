import numpy as np
from PIL import Image

from util import svd
from util import plots


ex_img_path = '/workspaces/workspace/res/cat.jpg'

img = Image.open(ex_img_path)
original_img = np.array(img)

svd_res = svd.compress_mc(original_img)

plots.compare_compressed(original_img, svd_res.compressed_img)
plots.scree_plot(svd_res.S1)