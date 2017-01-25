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
import pickle
#image  = image file, size = desired size of subregion (32, in our case), path = where to place subregions (optional)

def sample(train_batch,sy,sx,nl):
	sample = []
	for i in range(train_batch):
		temp0 = []
		for j in range(sy):
			temp1 = []
			for k in range(sx):
				temp1.append(range(i*sy*sx*nl+j*sx*nl+k*nl,i*sy*sx*nl+j*sx*nl+k*nl+nl))
			temp0.append(temp1)
		sample.append(temp0)
	return np.array(sample)

def get_segments(pool_size, pools):
	output_size = pool_size/pools
	segment_ids = np.zeros(pool_size)
	for i in range(pools,pool_size,pools):
		for pool in range(pools):
			segment_ids[i+pool] = int(i/pools)

	return np.int32(segment_ids)

def get_slice(image, size):
	img = cv2.imread(image)

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

		subregions["row %d" % (row)] = image_list

	return {"subregions": subregions, "dimensions": [rows, columns, length, width]}

def get_data_ae(source,classes,size):
	train_data = []
	out_val = len(classes)

	for cl in classes:
		f_source = source + cl
		for name in os.listdir(f_source):
			image = os.path.join(f_source,name)
			if os.path.isfile(image):
				img_list = get_slice(image, size)["subregions"]

				for row in img_list:
					for img in img_list[row]:
						train_data.append(img/float(255))

	return np.array(train_data)

def get_data_super(source,classes,size):
	train_data = []
	out_val = len(classes)

	for cl in classes:
		one_hot = np.zeros(out_val)
		one_hot[classes.index(cl)] = 1

		f_source = source + cl
		for name in os.listdir(f_source):
			image = os.path.join(f_source,name)
			if os.path.isfile(image):
				img_list = get_slice(image, size)["subregions"]

				for row in img_list:
					for img in img_list[row]:
						train_data.append((img/float(255),one_hot))

	return np.array(train_data)

#returns a pair (x,y) where x is input list, y is output list


def get_batch(data, num):
	x, y = zip(*random.sample(data,num))
	return x,y

def get_batch_grouped(data, num, out_val):
	x = []
	y = []
	class_len = len(data)/3
	for i in range(out_val):
		x_, y_ = zip(*random.sample(data[class_len*i:class_len*(i+1)],num))
		x.extend(x_)
		y.extend(y_)
	return x,y

def get_batch_x(data, num):
	return np.array(random.sample(data,num))

def mean_pool_2x2(in_np,train_batch,sy,sx,nf):
	x = np.transpose(in_np, (0,3,1,2))
	new_x = []
	for i in range(train_batch):
		temp0 = []
		for j in range(nf):
			temp1 = []
			for k in range(0,sy,2):
				temp2 = []
				for l in range(0,sx,2):
					temp2.append(np.mean(x[i,j,k:k+2,l:l+2]))
				temp1.append(temp2)
			temp0.append(temp1)
		new_x.append(temp0)
	return np.transpose(np.array(new_x), (0,2,3,1))


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_transpose(x, W, output_shape):
	return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='SAME')

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
    

def risa_segment_sum():
	segment_ids = tf.constant(get_segments(nl,4))
	x = tf.placeholder(tf.float32, shape = [train_batch,sy,sx,nl])
	x_tr = tf.transpose(x, perm = [0,3,1,2])
	y = tf.segment_sum(x_tr, segment_ids)
	y_fin = tf.transpose(y, perm = [0,2,3,1])

	return y_fin

def unpool_2x2(pooled,py,px,pf):
	a = tf.transpose(pooled, perm = [0,3,1,2])
	b = tf.reshape(a,[-1,pf,py*px,1])
	c = tf.tile(b,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
	d = tf.reshape(c,[-1,pf,py,px*2])
	e = tf.tile(d,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
	f = tf.reshape(e, [-1,pf,py*2,px*2])
	x_unpool = tf.transpose(f,perm = [0,2,3,1])
	return x_unpool

def showplot(input_rgb,output_rgb, sy, sx, train_batch, image_reco, in_type):
	ints = random.sample(range(train_batch), image_reco)
	
	#orig. images concatenated
	orig_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
	for i in range(image_reco):
		index = ints[i]
		orig_image[:,sx*i:sx*(i+1)] = input_rgb[index]
	
	#new images concatenated
	group = np.array([output_rgb[ints[0]]])
	
	for i in ints[1:]:
		group = np.concatenate((group,np.array([output_rgb[i]])), axis = 2)

	new_image = np.zeros((sy,sx*image_reco, 3), np.uint8)
	new_image[:,:] = group
	
	#plot
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.subplot(211)
	plt.title('Original Images ({}x{})'.format(sy,sx))
	plt.imshow(orig_image, 'gray')
	plt.xticks([]), plt.yticks([])

	plt.subplot(212)
	plt.title('Reconstructed Images ({}x{})'.format(sy,sx))
	plt.imshow(new_image, 'gray')
	plt.xticks([]), plt.yticks([])

	plt.savefig('Images/{}.png'.format(in_type))
	# plt.show()


def save_params(net,params_dict):
	for param in params_dict:
		in_text = open('Params/{}_{}.pkl'.format(net,param), 'wb')
		pickle.dump(params_dict[param], in_text)

def load_params(net,params_list):
	params_dict = {}
	for param in params_list:
		out_text = open('Params/{}_{}.pkl'.format(net,param), 'rb')
		params_dict[param] = tf.constant(pickle.load(out_text))
	return params_dict

def save_pickle(input_rgb, output_rgb):
	# x = pickle.load(open('test.pkl' ,'rb'))
	in_text = open('Sample/input_{}.pkl'.format(timestamp()), 'wb')
	pickle.dump(input_rgb, in_text)

	out_text = open('Sample/output_{}.pkl'.format(timestamp()), 'wb')
	pickle.dump(output_rgb, out_text)

def net_name(ae_type,sy,fs1,time):
	if time == 1:
		time_cur = datetime.datetime.now()
		return '{}_{}_{}_{}'.format(ae_type,sy,fs1, time_cur.strftime('%Y%m%d%H%M'))
	else:
		return '{}_{}_{}'.format(ae_type,sy,fs1)


def conf_matrix(y_predict,classes,class_batch):
	conf_matrix = {}
	out_val = len(classes)

	for i in range(out_val):
		conf_matrix[classes[i]] = [0,0,0]
		for j in y_predict[class_batch*i:class_batch*(i+1)]:
			conf_matrix[classes[i]][j] += 1

	return conf_matrix

def filters():
	dim = 2

	input_img = pickle.load(open( "Sample/test_in.pkl", "rb" ))
	filters = pickle.load(open( "Sample/test_out.pkl", "rb" ))

	n, sy, sx, nf = np.shape(filters)

	filters = np.transpose(filters[0],(2,0,1))
	fil_sample = random.sample(filters*255, dim**2)


	# plt.subplots_adjust(wspace=0, hspace=0)
	plt.figure()
	for i in range(dim):
		for j in range(dim):
			index = i*dim + j
			fil_image = np.zeros((sy,sx), np.uint8)
			fil_image[:,:] = np.array(fil_sample[index])
			print (dim**2,i+1,j+1)

			plt.subplot(dim**2,i+1,j+1)
			plt.title('Filter {}'.format(index))
			plt.imshow(fil_image, 'gray')
			plt.xticks([]), plt.yticks([])
	plt.show()