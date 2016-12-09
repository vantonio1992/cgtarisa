#standard packages

from PIL import Image
import numpy as np
import os
import timeit
import pickle, random
import datetime
from functions import *
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf


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
                # train_data.append(get_layered_rgb(img))
                train_data.append(img)

train_data = np.array(train_data)


#getting sample (not yet /255)
input_rgb = np.array(random.sample(train_data,image_reco))

input_tf = input_rgb/float(255)


#network

keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
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

#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()


#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("(verification of {} with {}x{} input)".format('CNN_ae',sy,sx))
saver.restore(sess, "{}_weights.ckpt".format('CNN_ae'))
output_rgb = h_deconv2.eval(feed_dict={x_image: input_tf})
print('end')
showplot(input_rgb,output_rgb, sy, sx, image_reco)

