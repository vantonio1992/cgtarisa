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


def get_rgb(image):
    img = cv2.imread(image)
    
    length = img.shape[0]
    width = img.shape[1]
    classes = length*width
    blue_vec=np.tile([0],classes)
    green_vec=np.tile([0],classes)
    red_vec=np.tile([0],classes)
    
    for row in range(length):
        for column in range(width):
            blue_value = img[row,column,0]
            green_value = img[row,column,1]
            red_value = img[row,column,2]
            
            blue_vec[row*width+column] = blue_value
            green_vec[row*width+column] = green_value
            red_vec[row*width+column] = red_value

            
    #return [length, width]
    return np.append(blue_vec, [green_vec, red_vec])

def timestamp():
    time_cur = datetime.datetime.now()
    print('datetime:',time_cur.strftime('%m/%d %H:%M'))
    stamp = time_cur.strftime('%Y%m%d%H%M')
    return(stamp)