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

#parameters, manual input
source = "Training"

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

#model implementation



#Training start

#first conv. layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])

x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second conv. layer
W_conv2 = weight_variable([fs2, fs2, nf1, nf2])
b_conv2 = bias_variable([nf2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#reverse the process

##unpool, deconv 1

#unpooling
a1 = tf.transpose(h_pool2, perm = [0,3,1,2])
b1 = tf.reshape(a1,[-1,nf2,sy/4*sx/4,1])
c1 = tf.tile(b1,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
d1 = tf.reshape(c1,[-1,nf2,sy/4,sx/2])
e1 = tf.tile(d1,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
h_unpool1 = tf.reshape(e1, [-1,sy/2,sx/2,nf2])

#deconv
W_conv2_tr = tf.transpose(W_conv2, perm = [0,1,3,2])
h_deconv1 = tf.nn.relu(conv2d(h_conv2 - b_conv2, W_conv2_tr))

##unpool, deconv 2

#unpooling
a2 = tf.transpose(h_deconv1, perm = [0,3,1,2])
b2 = tf.reshape(a2,[-1,nf1,sy/2*sx/2,1])
c2 = tf.tile(b2,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
d2 = tf.reshape(c2,[-1,nf1,sy/2,sx])
e2 = tf.tile(d2,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
h_unpool2 = tf.reshape(e2, [-1,sy,sx,nf1])

#deconv
W_conv1_tr = tf.transpose(W_conv1, perm = [0,1,3,2])
h_deconv2 = tf.nn.relu(conv2d(h_unpool2 - b_conv1, W_conv1_tr))


#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)



saver.restore(sess, "Weights/{}_weights.ckpt".format('CNN_super'))

print("(verification of {} with {}x{} input)".format('CNN_super',sy,sx))
input_tf = np.array(random.sample(train_data,image_reco))
input_rgb = input_tf*float(255)
output_rgb = h_deconv2.eval(feed_dict={x_image: input_tf})
print('end')
showplot(input_rgb,output_rgb, sy, sx, image_reco)