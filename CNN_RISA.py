#standard packages

from PIL import Image

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

#general placeholders
keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])

#conv and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
# b_conv1 = bias_variable([nf1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
h_pool1 = max_pool_2x2(h_conv1)

##layers for RISA

#square
W_risa = weight_variable([sy/2,sx/2,nf1,nf2])
W_t = tf.transpose(W_risa, perm = [0,1,3,2])
risa_pre_sq = tf.nn.relu(conv2d(h_pool1, W_risa))
risa_sq = tf.square(risa_pre_sq)

#sqrt
risa_pre_tf = tf.transpose(risa_sq, perm = [0,3,1,2])

risa_post_tf = tf.placeholder(tf.float32, [None,nf2,sy/4,sx/4])
risa_root = tf.sqrt(risa_post_tf)

#error check

output_t = tf.nn.relu(conv2d(risa_pre_sq, W_t))
# output_drop = tf.nn.dropout(output_t, keep_prob)

norm = tf.reduce_mean(tf.square(tf.global_norm([tf.sub(output_t,h_pool1)])))
error = tf.add(tf.reduce_sum(risa_root),tf.mul(tf.to_float(tf.constant(lambda_r)),norm))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)
#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("RISA Network with {}x{} input".format(sy,sx))

error_list = []
for i in range(maxiter):
    batch_xs, batch_ys = get_batch(train_data,train_batch)
    if i%moditer == 0:
        pre_root = risa_pre_tf.eval(feed_dict = {x_image: batch_xs})
        post_root = get_sum_2x2(pre_root,train_batch,nf2,sy/2,sx/2)
        train_error = error.eval(feed_dict = {x_image: batch_xs, risa_post_tf: post_root})
        error_list.append(train_error)
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, keep_prob: 0.5})


error_file = open("{}_error_{}x{}.txt".format("RISA",sy,sx), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close