#standard packages

from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *

def multi_showplot(input_rgb, output_rgb_dict, train_batch, image_reco, sy, sx):
	ints = random.sample(range(train_batch), image_reco)
	length = len(output_rgb_dict)
	#orig. images concatenated
	orig_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
	for i in range(image_reco):
		index = ints[i]
		orig_image[:,sx*i:sx*(i+1)] = input_rgb[index]
	
	#new images concatenated
	new_image_dict = {}
	for in_type in output_rgb_dict:
		output_rgb = output_rgb_dict[in_type]

		group = np.array([output_rgb[ints[0]]])
		
		for i in ints[1:]:
			group = np.concatenate((group,np.array([output_rgb[i]])), axis = 2)

		new_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
		new_image[:,:] = group*255
		new_image_dict[in_type] = new_image
	
	# plot
	plt.subplot(3,1,1)
	plt.title('Original Images ({}x{})'.format(sy,sx))
	plt.imshow(orig_image, 'gray')
	plt.xticks([]), plt.yticks([])

	i = 0
	in_type = output_rgb_dict.keys()[i]
	plt.subplot(3,2,1)
	plt.title('Reconstructed {}x{} Images ({})'.format(sy,sx,in_type))
	plt.imshow(new_image_dict[in_type], 'gray')
	plt.xticks([]), plt.yticks([])

	i = 1
	in_type = output_rgb_dict.keys()[i]
	plt.subplot(3,3,1)
	plt.title('Reconstructed {}x{} Images ({})'.format(sy,sx,in_type))
	plt.imshow(new_image_dict[in_type], 'gray')
	plt.xticks([]), plt.yticks([])
	# time_cur = datetime.datetime.now()
	# timestamp = time_cur.strftime('%Y%m%d%H%M')
	# plt.savefig('Images/{}_{}.png'.format('make_plot',timestamp))
	plt.show()

exec(open('extern_params.py').read())

train_dict = {}
train_data = []

for image in images:
    for n in range(train_f):
        train_list = get_slice('{}/{}'.format(training, image), '{}{}.jpeg'.format(image,n), sx)
        train_dict['{}{}'.format(image,n)] = train_list["subregions"]

        for row in train_dict['{}{}'.format(image,n)]:
            for img in train_dict['{}{}'.format(image,n)][row]:
                train_data.append(img/float(255))

train_data = np.array(train_data)

input_tf = get_batch_x(train_data,train_batch)
input_rgb = input_tf*255

output_rgb_dict = {}
exec(open('test_CNN_ae.py').read())
output_rgb_dict['test_CNN_ae'] = output_rgb
exec(open('test_CNN_Hybrid.py').read())
output_rgb_dict['test_CNN_Hybrid'] = output_rgb

multi_showplot(input_rgb, output_rgb_dict, train_batch, image_reco, sy, sx)
