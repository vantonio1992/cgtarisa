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

net = '{}_{}_{} ({})'.format('CNN_Hybrid',sy,fs1,timestamp())

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
# b_conv1 = bias_variable([nf1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1))


h_conv1_tr = tf.square(tf.transpose(h_conv1, perm = [0,3,1,2]))

h_pre_sqrt = tf.placeholder(tf.float32, shape = [train_batch,nf1,sy/2,sx/2])
h_sqrt = tf.sqrt(h_pre_sqrt)
h_pooled = tf.transpose(h_sqrt, perm = [0,2,3,1])



unpool1 = unpool_2x2(h_pooled,sy/2,sx/2,nf1)
output_shape1 = [train_batch,sy,sx,nl]
h_deconv1 = tf.nn.relu(conv2d_transpose(unpool1,W_conv1,output_shape1))

norm = tf.reduce_mean(tf.global_norm([h_deconv1 - x_image]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(norm)
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
        h_conv1_np = h_conv1_tr.eval(feed_dict = {x_image: batch_xs})
        pre_sqrt = get_sum_2x2(h_conv1_np,train_batch,nf1,sy,sx)
        train_error = norm.eval(feed_dict = {x_image: batch_xs, h_pre_sqrt: pre_sqrt, keep_prob: 1.0})
        error_list.append(train_error)
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, h_pre_sqrt: pre_sqrt, keep_prob: 0.5})


error_file = open("Errors/error_{}.txt".format(net), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close



saver.save(sess, "Weights/weights_{}.ckpt".format(net))

#confirm plotting
print('start reconstruction')
input_tf = get_batch_x(train_data,train_batch)
h_conv1_np = h_conv1_tr.eval(feed_dict = {x_image: input_tf})
pre_sqrt = get_sum_2x2(h_conv1_np,train_batch,nf1,sy,sx)
input_rgb = input_tf*255
output_rgb = h_deconv1.eval(feed_dict={x_image: input_tf, h_pre_sqrt: pre_sqrt})
showplot(input_rgb,output_rgb, sy, sx, train_batch, image_reco, net)

