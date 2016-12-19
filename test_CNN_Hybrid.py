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
net = net_name('CNN_Hybrid',sy,fs1,time)


#start of implementation


#Training start

#general placeholders
keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, shape = [None,sy,sx,nl])

##conv, pooling, conv

#conv1
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))


#pool
h_pool = tf.placeholder(tf.float32, shape = [None,sy/2,sx/2,nf1])
unpool1 = unpool_2x2(h_pool,sy/2,sx/2,nf1)
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
    saver.restore(sess, "Weights/weights_{}.ckpt".format(net))


saver.save(sess, "Weights/weights_{}.ckpt".format(net))

#confirm plotting
print('start reconstruction')
h_pre_mean = h_conv1.eval(feed_dict = {x_image: input_tf})
pre_pool = mean_pool_2x2(h_pre_mean,train_batch,sy,sx,nf1)
output_rgb = h_deconv1.eval(feed_dict={x_image: input_tf, h_pool: pre_pool})

