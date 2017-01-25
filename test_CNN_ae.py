#standard packages

from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *


exec(open('extern_params.py').read())

#gathering data from images

net = net_name('CNN_ae',sy,fs1,time)

train_data = get_data_ae(training,classes,sy)


#start of implementation


#Training start

#general placeholders
keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])

##conv, pooling, conv

#conv1 and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#conv2 and pooling layer
W_conv2 = weight_variable([fs2, fs2, nf1, nf2])
b_conv2 = bias_variable([nf2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#unpool, deconv2
unpool2 = unpool_2x2(h_pool2,sy/4,sx/4,nf2)
output_shape2 = [train_batch,sy/2,sx/2,nf1]
h_deconv2 = tf.nn.relu(conv2d_transpose(unpool2,W_conv2,output_shape2))

#unpool, deconv1
unpool1 = unpool_2x2(h_deconv2,sy/2,sx/2,nf1)
output_shape1 = [train_batch,sy,sx,nl]
h_deconv1 = tf.nn.relu(conv2d_transpose(unpool1,W_conv1,output_shape1))


#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()


#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("({} test)".format(net,sy,sx))

if switch == 1:
    saver.restore(sess, "Weights/{}_weights.ckpt".format(net))


saver.save(sess, "Weights/{}_weights.ckpt".format(net))

#confirm plotting
print('start reconstruction')

input_tf = get_batch_x(train_data,train_batch)
input_rgb = input_tf*255
output_tf = h_deconv1.eval(feed_dict={x_image: input_tf})
showplot(input_rgb,output_tf*255, sy, sx, train_batch, image_reco, net)

