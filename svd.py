import numpy as np
from PIL import Image
from scipy import linalg


ex_img_path = '/workspaces/workspace/res/cat.jpg'

image = Image.open(ex_img_path)

image.show()