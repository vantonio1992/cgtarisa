#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train risa')

import os
import sys
import csv
import numpy as np
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
from functions import *

exec(open('extern_params.py').read())



#
# load sample data
#


#get sample input

img_dict = {}
for image in images:
	img_list = get_slice(training, "%s.jpeg" % (image), input_size)
	img_dict[image] = img_list["subregions"]

data_set = []
for name in images:
	for row in img_dict[name]:
		for img in img_dict[name][row]:
			data_set.append(get_rgb(img))

data_set = np.array(data_set)


input_tf = tf.Variable(tf.random_normal([2,ny,nx,nl]))
filter_tf = tf.Variable(tf.random_normal([fs,fs,nl,nf]))

#
# setup optimizer
#

#define network
segment_ids = tf.constant(get_segments(pool_size,pools))

risa_in =  tf.placeholder(tf.float32, shape = [None,ny,nx,nl])
conv_input = tf.nn.conv2d(risa_in, filter_tf, strides=[1, sl, sl, 1], padding='SAME')
pool_vec = tf.nn.max_pool(tf.square(tf.matmul(W_matrix,conv_input)), ksize = [1,2,2,1], strides =  [1,2,2,1], padding = 'SAME')
output_vec = tf.sqrt(tf.segment_sum(pool_vec, segment_ids))


#variables

W_matrix = tf.Variable(tf.zeros([input_size,pool_size]))
w_w_transpose = tf.matmul(W_matrix,tf.transpose(W_matrix))


#
#error gathering
#

invariance = tf.reduce_sum(output_vec)
norm = tf.sub(tf.matmul(risa_in,w_w_transpose),risa_in)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
#train = optimizer.minimize(sum_invariance + lambda_s*mean_error)



#start interactive session (place at the end)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())




#
#computation start
#


#for invariance
inv_set = []
for data in data_set:
	pool_vec = tf.square(tf.matmul(risa_in,W_matrix)[0, :])
	risa_out = tf.sqrt(tf.to_float(tf.segment_sum(pool_vec, segment_ids)))
	inv = tf.reduce_sum(risa_out)

	inv_set.append(sess.run(inv,feed_dict = {risa_in: np.array([data])}))
inv_set = np.array(inv_set)






			
