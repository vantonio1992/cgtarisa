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


#pool_vec is tensor, pool_size and pool are int
def V_output_tensor(pool_vec,pool_size,pools):
	sess = tf.InteractiveSession()
	x_vec = tf.placeholder()
	segment_ids = np.zeros(pool_size)
	for i in range(pools,pool_size,pools):
		for pool in range(pools):
			segment_ids[i+pool] = int(i/pools)

	result = tf.segment_sum(pool_vec)

	result = sess.run(tf.segment_sum(pool_vec,segment_ids))

	final = np.sqrt(result)
	return final


#W is tf.matrix, input_np is standard row vector
#W_matrix = tf.to_float(W)
def W_output(input_np,W_np):

	result = input_np.dot(W_np)
	W_output_square = np.square(result)
	return W_output_square

#input is np.array



#pool_vec is array, pool_size and pools are int
def risa_output(input_np,W_np,pool_size,pools):
	pool_vec = np.square(input_np.dot(W_np))

	output_size = pool_size/pools
	segment_ids = [0]*output_size
	for i in range(len(segment_ids)):
		segment_ids[i] = i*pools

	result = np.add.reduceat(pool_vec,segment_ids)
	V_output_sqrt = np.sqrt(result)
	
	return V_output_sqrt
	#output is array

def get_segments(pool_size,pools):
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


def feature(image):
    img = cv2.imread(image)
    
    length = img.shape[0]
    width = img.shape[1]
    classes = int(math.ceil(math.log(length*width+1,2)))
    class_width = int(math.ceil(255/float(classes)))
    
    blue_histogram=np.tile([0],classes)
    green_histogram=np.tile([0],classes)
    red_histogram=np.tile([0],classes)
    
    for row in range(length):
        for column in range(width):
            blue_value = img[row,column,0]
            green_value = img[row,column,1]
            red_value = img[row,column,2]
            
            blue_histogram[int(blue_value/float(class_width))] += 1
            green_histogram[int(green_value/float(class_width))] += 1
            red_histogram[int(red_value/float(class_width))] += 1

            
    #return [length, width]
    return np.append(blue_histogram, [green_histogram, red_histogram])


def get_rgb(img):
	#img = cv2.imread(image)

	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]

	#return [length, width]
	return np.array([b,g,r])/255

def get_layered_rgb(img):
	length = img.shape[0]
	width = img.shape[1]

	rgb_list = []
	for row in range(length):
		row_np = np.empty([width,3])
		for col in range(width):
			row_np[col][0] = img[row,col,0]
			row_np[col][1] = img[row,col,1]
			row_np[col][2] = img[row,col,2]
		rgb_list.append(row_np)

	return np.array(rgb_list)/255


#returns a pair (x,y) where x is input list, y is output list
def get_batch(data, num):
	x, y = zip(*random.sample(data,num))
	return x,y


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