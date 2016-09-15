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
import functions as *

exec(open('extern_params.py').read())



#start session
sess = tf.InteractiveSession()

#define W and V from RISA

W_risa = tf.Variable(tf.zeros([pool_size,input_size]))

#
# load sample data
#

# sample size
input_size = 64
pool_size = 32
output_size = 32

file_input = 'risa_trn_w{}.npy'.format(ss)
path_data = os.path.join(dir_input,'input_w{}'.format(ss),file_input)
risa_trn = np.load(path_data)
print('load input from {}'.format(path_data))

nn,ny,nx,nl = risa_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)



#get sample input

img_dict = {}
for image in images:
	img_list = get_slice()


#
# setup optimizer
#

#define input
risa_in =  tf.placeholder(tf.float32)
pool_vec = tf.placeholder(tf.float32)
W_matrix = tf.placeholder(tf.float32)
segment_ids = tf.constant(get_segments(pool_size,pools))

#variables

W_matrix = tf.Variable(tf.zeros([input_size,pool_size]))

sess.run(tf.initialize_all_variables())



invariance = tf.reduce_sum(risa_output)
mean_error = tf.reduce_mean(tf.square(risa_deconv1 - risa_input))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error + lambda_s*mean_entropy)



#start interactive session (place at the end)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())