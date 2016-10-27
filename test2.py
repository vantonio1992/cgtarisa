import random, cv2
import numpy as np
from functions import *

#exec(open('extern_params.py').read())

sy = 6
sx = 4

rgb_list = []
for i in range(2):
	img = cv2.imread('train{}.jpeg'.format(i))
	rgb_list.append(get_layered_rgb(img))


print np.array(rgb_list).shape











