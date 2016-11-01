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
    one_hot = np.zeros(3)
    one_hot[images.index(image)] = 1

    for n in range(train_f):
        train_list = get_slice('{}/{}'.format(training, image), '{}{}.jpeg'.format(image,n), sx)
        train_dict['{}{}'.format(image,n)] = train_list["subregions"]

        for row in train_dict['{}{}'.format(image,n)]:
            for img in train_dict['{}{}'.format(image,n)][row]:
                train_data.append((get_layered_rgb(img),one_hot))

print len(train_data)



#start of implementation

#weights, bias incorporation
y_ = tf.placeholder(tf.float32, shape=[None, out_val])

W = tf.Variable(tf.zeros([input_size, out_val]))
b = tf.Variable(tf.zeros([out_val]))

#model implementation



#Training start

#first conv. layer
W_conv1 = weight_variable([fs1, fs1, nl, nf1])
b_conv1 = bias_variable([nf1])

x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)




#densely-connected layer
W_fc1 = weight_variable([output_y * output_x * nf2, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, output_y*output_x*nf2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#readout layer

W_fc2 = weight_variable([1024, out_val])
b_fc2 = bias_variable([out_val])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

#learning rate = 0.01
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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