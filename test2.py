import random, cv2
import numpy as np
from functions import *

#exec(open('extern_params.py').read())

exec(open('extern_params.py').read())

test_dict = {}
test_data = []

for image in images:
	one_hot = np.zeros(3)
	one_hot[images.index(image)] = 1

	for n in range(test_f):
		test_list = get_slice('{}/{}'.format(testing, image), '{}{}.jpeg'.format(image,n), sx)
		test_dict['{}{}'.format(image,n)] = test_list["subregions"]

		for row in test_dict['{}{}'.format(image,n)]:
			for img in test_dict['{}{}'.format(image,n)][row]:
				test_data.append((get_layered_rgb(img),one_hot))

print test_data[0][1]



