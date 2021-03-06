#standard packages

from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *



#network

net = 'CNN_ae_bias'

#gathering data from images
exec(open('extern_params.py').read())

train_dict = {}
train_data = []

for image in images:
    for n in range(train_f):
        train_list = get_slice('{}/{}'.format(training, image), '{}{}.jpeg'.format(image,n), sx)
        train_dict['{}{}'.format(image,n)] = train_list["subregions"]

        for row in train_dict['{}{}'.format(image,n)]:
            for img in train_dict['{}{}'.format(image,n)][row]:
                train_data.append(img/float(255))

train_data = np.array(train_data)


#start of implementation


#Training start

#general placeholders
keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])

##conv, pooling, conv

#conv1 and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])
W_conv1_tr = tf.transpose(W_conv1, perm = [0,1,3,2])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#conv2
W_conv2 = weight_variable([sy/2,sx/2,nf1,nf2])
b_conv2 = bias_variable([nf2])
W_conv2_tr = tf.transpose(W_conv2, perm = [0,1,3,2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#deconv, unpooling, deconv
unpool2 = unpool_2x2(h_pool2,sy/4,sx/4,nf2)
h_deconv2 = tf.nn.relu(conv2d(h_conv2 - b_conv2, W_conv2_tr))
unpool1 = unpool_2x2(h_deconv2,sy/2,sx/2,nf1)
h_deconv1 = tf.nn.relu(conv2d(unpool1 - b_conv1, W_conv1_tr))

norm = tf.reduce_mean(tf.global_norm([tf.sub(h_deconv1,x_image)]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(norm)
#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()


#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("({} with {}x{} input)".format(net,sy,sx))

if switch == 1:
    saver.restore(sess, "Weights/{}_weights.ckpt".format(net))

error_list = []
for i in range(maxiter):
    batch_xs = get_batch_x(train_data,train_batch)
    if i%moditer == 0:
        train_error = norm.eval(feed_dict = {x_image: batch_xs, keep_prob: 1.0})
        error_list.append(train_error)
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, keep_prob: 0.5})


error_file = open("Errors/{}_error_{}x{}.txt".format(net,sy,sx), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close

saver.save(sess, "Weights/{}_weights.ckpt".format(net))

#confirm plotting
print('start reconstruction')
input_tf = get_batch_x(train_data,image_reco)
input_rgb = input_tf*float(255)
output_rgb = h_deconv1.eval(feed_dict={x_image: input_tf})
showplot(input_rgb,output_rgb, sy, sx, image_reco)