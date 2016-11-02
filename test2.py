import random, cv2
import numpy as np
from functions import *


exec(open('extern_params.py').read())

img = cv2.imread('train0.jpeg')

print get_layered_rgb(img)
