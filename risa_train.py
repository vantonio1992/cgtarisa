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
import tensorflow_ae_base

from tensorflow_ae_base import *
import tensorflow_util

import functions

exec(open('extern_params.py').read())

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



#
# setup optimizer
#
risa_input = tf.placeholder(tf.float32, [None,ny,nx,nl])


#change later
risa_output = tf.placeholder(tf.float32, [None,ny,nx,nl])


#variables

risa_square = tf.Variable(tf.zeros([input_size,simple_size]))

risa_root = tf.constant(tf.zeros([pool_size,output_size]))

risa_encode1 = get_encode1(risa_input)
risa_deconv1 = get_deconv1(risa_encode1)
mean_error = tf.reduce_mean(tf.square(risa_deconv1 - risa_input))
local_entropy = get_local_entropy_encode1(risa_encode1)
mean_entropy = tf.reduce_mean(local_entropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error + lambda_s*mean_entropy)



#start interactive session (place at the end)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())