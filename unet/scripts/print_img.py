from PIL import Image
import numpy as np
from scipy.misc import imsave

IMAGE = 'predicted_mask_2'

img = Image.open(IMAGE+".png")
img = np.array(img)

print(np.unique(img))

img = img * 255.0

imsave(IMAGE+"_img.png", img)
