#standard packages

from PIL import Image

import csv
import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *



#gathering data from images
exec(open('extern_params.py').read())

train_dict = {}
train_data = []

for image in images:
    one_hot = np.zeros(3)
    one_hot[images.index(image)] = 1

    for n in range(train_f):
        train_list = get_slice('{}/{}'.format(training, image), '{}{}.jpeg'.format(image,n), sx)
        train_dict['{}{}'.format(image,n)] = train_list["subregions"]

        for row in train_dict['{}{}'.format(image,n)]:
            for img in train_dict['{}{}'.format(image,n)][row]:
                train_data.append((get_layered_rgb(img),one_hot))

train_data = np.array(train_data)


#start of implementation


#Training start

#conv and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])

x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

##layers for RISA

#square
risa_in = tf.square(h_pool1)
W_risa = weight_variable([sy/2,sx/2,nf1,nf1])
risa_sq = tf.nn.relu(conv2d(risa_in, W_risa))
risa_sq_flat = tf.reshape(risa_sq, [-1,nf1,sy/2*sx/2])

#sqrt
risa_pre_tf = tf.placeholder(tf.float32, shape=[nf1, sy/2*sx/2])
segment_ids = tf.constant(get_segments(nf1,risa_pool))
risa_root = tf.sqrt(tf.segment_sum(risa_pre_tf, segment_ids))

#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#W transpose
W_risa_t = []

W_risa_np = W_risa.eval()
for a in W_risa_np:
    temp = []
    for b in a:
        temp.append(np.transpose(b))
    W_risa_t.append(temp)

W_risa_t = tf.constant(np.array(W_risa_t))

#risa 2nd layer
batch_xs, batch_ys = get_batch(train_data,train_batch)

risa_sq_np = risa_sq_flat.eval(feed_dict = {x_image: batch_xs})

risa_sqrt = []
for img in risa_sq_np:
    risa_sqrt.append(risa_root.eval(feed_dict = {risa_pre_tf: img}))

risa_sqrt = np.array(risa_sqrt)


#cross_entropy
output = tf.nn.relu(conv2d(h_pool1, W_risa))
output_t = tf.nn.relu(conv2d(output, W_risa_t))
norm = tf.global_norm([tf.sub(output_t,h_pool1)])

print norm.eval(feed_dict = {x_image: batch_xs})
