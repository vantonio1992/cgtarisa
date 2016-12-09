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

net = '{}_{}_{} ({})'.format('RISA',sy,fs1,timestamp())

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
W_risa1 = weight_variable([fs1, fs1, nl, nf1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_risa1))

#RISA

risa_sq = tf.square(h_conv1)

#sqrt
segment_ids = tf.constant(get_segments(nf1,4))
risa_sq_tr = tf.transpose(risa_sq, perm = [3,0,1,2])
risa_sum = tf.segment_sum(risa_sq_tr, segment_ids)
risa_root = tf.sqrt(risa_sum)


#reverse process

#unpool
a = tf.transpose(risa_root, perm = [1,2,3,0])
b = tf.reshape(a, [train_batch,sy,sx*nf1/4,1])
c = tf.tile(b,tf.to_int32(tf.constant(np.array([1,1,1,4]))))
risa_unpool = tf.reshape(c, [train_batch,sy,sx,nf1])

## error check
# deconv_compute

output_shape1 = [train_batch,sy,sx,nl]
h_deconv_c = tf.nn.relu(conv2d_transpose(h_conv1,W_risa1,output_shape1))

norm = tf.reduce_mean(tf.global_norm([h_deconv_c - x_image]))
error = tf.add(tf.reduce_sum(risa_root),tf.mul(tf.to_float(tf.constant(lambda_r)),norm))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)


#deconv_ae
h_deconv_ae = tf.nn.relu(conv2d_transpose(risa_unpool,W_risa1,output_shape1))


#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()


#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("({} simulation)".format(net,sy,sx))

if switch == 1:
    saver.restore(sess, "Weights/{}_weights.ckpt".format(net))


error_list = []
for i in range(maxiter):
    batch_xs = get_batch_x(train_data,train_batch)
    if i%moditer == 0:
        train_error = error.eval(feed_dict = {x_image: batch_xs, keep_prob: 1.0})
        error_list.append(train_error)
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, keep_prob: 0.5})


error_file = open("Errors/error_{}.txt".format(net), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close



saver.save(sess, "Weights/weights_{}.ckpt".format(net))

#confirm plotting
print('start reconstruction')
input_tf = get_batch_x(train_data,train_batch)
input_rgb = input_tf*255
output_rgb = h_deconv_ae.eval(feed_dict={x_image: input_tf})
showplot(input_rgb,output_rgb, sy, sx, train_batch, image_reco, net)