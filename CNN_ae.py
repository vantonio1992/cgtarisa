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

##conv, pooling, conv

#conv1 and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
# b_conv1 = bias_variable([nf1])
W_conv1_tr = tf.transpose(W_conv1, perm = [0,1,3,2])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))
h_pool1 = max_pool_2x2(h_conv1)

#conv2
W_conv2 = weight_variable([sy/2,sx/2,nf1,nf2])
W_conv2_tr = tf.transpose(W_conv2, perm = [0,1,3,2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2))


##deconvolution, unpooling
h_deconv1 = tf.nn.relu(conv2d(h_conv2, W_conv2_tr))

#unpooling
a = tf.transpose(h_deconv1, perm = [0,3,1,2])
b = tf.reshape(a,[-1,nf1,sy/2*sx/2,1])
c = tf.tile(b,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
d = tf.reshape(c,[-1,nf1,sy/2,sx])
e = tf.tile(d,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
x_unpool = tf.reshape(e, [-1,sy,sx,nf1])


#deconv 2
h_deconv2 = tf.nn.relu(conv2d(x_unpool, W_conv1_tr))
# h_deconv2_drop = tf.nn.dropout(h_deconv2, keep_prob)

norm = tf.reduce_mean(tf.global_norm([tf.sub(h_deconv2,x_image)]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(norm)
#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("(autoencoder with {}x{} input)".format(sy,sx))

error_list = []
for i in range(maxiter):
    batch_xs, batch_ys = get_batch(train_data,train_batch)
    if i%moditer == 0:
        train_error = norm.eval(feed_dict = {x_image: batch_xs, keep_prob: 1.0})
        error_list.append(train_error)
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, keep_prob: 0.5})


error_file = open("{}_error_{}x{}.txt".format("CNN_ae",sy,sx), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close