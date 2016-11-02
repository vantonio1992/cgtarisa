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

#weights, bias incorporation
y_ = tf.placeholder(tf.float32, shape=[None, out_val])


#Training start

#conv and pooling layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])

x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layers for RISA
risa_sq = tf.reshape(tf.square(h_pool1), [-1,sy/2*sx/2,nf1])

W = tf.Variable(tf.zeros([sy/2*sx/2*nf2, sy/2*sx/2]))

segment_ids = tf.constant(get_segments(nf1,risa_pool))
risa_root = tf.sqrt(tf.segment_sum(tf.matmul(W,risa_sq), segment_ids))


#cross_entropy
output = tf.matmul(tf.matmul(tf.transpose(W),W),h_pool1)
norm = tf.square(tf.global_norm(tf.sub(output,h_pool1)))
error = tf.add(tf.reduce_sum(risa_root),tf.reduce_sum(tf.reduce_mean(norm)))
#learning rate = 0.01
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)

#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)



#input data here, read training data
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(1000):
    batch_xs, batch_ys = get_batch(train_data,train_batch)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob: 0.5})

#cross-validation
test_dict = {}
test_data = []

for image in images:
    one_hot = np.zeros(3)
    one_hot[images.index(image)] = 1

    for n in range(test_f):
        test_list = get_slice('{}/{}'.format(testing, image), '{}{}.jpeg'.format(image,n), sx)
        test_dict['{}{}'.format(image,n)] = test_list["subregions"]

        for row in test_dict['{}{}'.format(image,n)]:
            for img in test_dict['{}{}'.format(image,n)][row]:
                test_data.append((get_layered_rgb(img),one_hot))

test_data = np.array(test_data)

x_test, y_test = get_batch(test_data,test_batch)
print("test accuracy %g"%accuracy.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0}))