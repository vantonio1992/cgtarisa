import numpy as np
import timeit
import random
import decimal
import timeit
import cv2
from matplotlib import pyplot as plt
import os, math, datetime
import numpy as np
import tensorflow as tf
#image  = image file, size = desired size of subregion (32, in our case), path = where to place subregions (optional)


def get_segments(pool_size, pools):
	output_size = pool_size/pools
	segment_ids = np.zeros(pool_size)
	for i in range(pools,pool_size,pools):
		for pool in range(pools):
			segment_ids[i+pool] = int(i/pools)

	return np.int32(segment_ids)


def get_slice(source, image, size):
	img = cv2.imread("%s/%s" % (source,image))
	ctr = 0
	
	if not os.path.exists("%s/%dpx" % (source, size)):
		os.makedirs("%s/%dpx" % (source, size))
		os.makedirs("%s/%dpx/%s" % (source, size, image))
		ctr = 1
	else:
		if not os.path.exists("%s/%dpx/%s" % (source, size, image)):
			os.makedirs("%s/%dpx/%s" % (source, size, image))
			ctr = 1 #newly made folder

	length = img.shape[0]
	width = img.shape[1]

	rows = int(img.shape[0]/size)
	columns = int(img.shape[1]/size)
	subregions = {}

	for row in range(rows):
		image_list = []

		for column in range(columns):
			new_image = img[(row*size):((row+1)*size),(column*size):((column+1)*size)]
			image_list.append(new_image)

			if ctr == 1:
				#where to place subregions
				cv2.imwrite("%s/%dpx/%s/sr(%d,%d).jpeg" % (source, size, image, row, column), new_image)

		subregions["row %d" % (row)] = image_list

	return {"subregions": subregions, "dimensions": [rows, columns, length, width]}




#returns a pair (x,y) where x is input list, y is output list
def get_batch(data, num):
	x, y = zip(*random.sample(data,num))
	return x,y

def get_batch_x(data, num):
	x = np.array(random.sample(data,num))
	return x

def get_sum_2x2(x,train_batch,nf,sy,sx):
	new_x = []
	for i in range(train_batch):
		temp0 = []
		for j in range(nf):
			temp1 = []
			for k in range(0,sy,2):
				temp2 = []
				for l in range(0,sx,2):
					temp2.append(np.sum(x[i,j,k:k+2,l:l+2]))
				temp1.append(temp2)
			temp0.append(temp1)
		new_x.append(temp0)
	return np.array(new_x)



def get_risa_segments(sy,sx):
	a = np.arange(sy/2*sx/2).reshape((sy/2*sx/2,1))
	b = np.tile(a, 2).reshape(2,sx)
	return np.tile(b,2).reshape(sy,sx).flatten()


def deconv2d(x, W, output_shape):
  return tf.nn.conv2d_transpose(x, W, output_shape = tf.constant(np.array(output_shape)), strides=[1, 1, 1, 1], padding='SAME')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  

def timestamp():
    time_cur = datetime.datetime.now()
    print('datetime:',time_cur.strftime('%m/%d %H:%M'))
    stamp = time_cur.strftime('%Y%m%d%H%M')
    return(stamp)

#input size = output size
def showplot(input_rgb,output_rgb, sy, sx, image_reco):

	#orig. images concatenated
	orig_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
	for i in range(image_reco):
		orig_image[:,sx*i:sx*(i+1)] = input_rgb[i]
	
	#new images concatenated
	group = np.array([output_rgb[0]])
	
	for img in output_rgb[1:]:
		group = np.concatenate((group,np.array([img])), axis = 2)

	new_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
	new_image[:,:] = group*255
	
	#plot
	plt.subplot(211)
	plt.title('Original Images ({}x{})'.format(sy,sx))
	plt.imshow(orig_image, 'gray')
	plt.xticks([]), plt.yticks([])

	plt.subplot(212)
	plt.title('New Images ({}x{})'.format(sy,sx))
	plt.imshow(new_image, 'gray')
	plt.xticks([]), plt.yticks([])

	plt.show()