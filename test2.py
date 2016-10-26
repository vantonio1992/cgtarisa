import random, cv2
import numpy as np
from functions import *

exec(open('extern_params.py').read())

img_dict = {}
for image in images:
    img_list = get_slice(training, "%s.jpeg" % (image), input_size)
    img_dict[image] = img_list["subregions"]

data_set = []
for name in images:
    for row in img_dict[name]:
        for img in img_dict[name][row]:
            data_set.append((get_rgb(img).flatten(),images.index(name)))

data_set = np.array(data_set)

print type(get_batch(data_set,3))


