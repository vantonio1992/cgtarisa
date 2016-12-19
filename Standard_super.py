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

net = net_name('Standard_super',sy,fs1,time)

train_data = get_data_super(training,classes,sy)



#start of implementation


#Training start

#general placeholders
keep_prob = tf.placeholder(tf.float32)
x_image = tf.placeholder(tf.float32, [None,sy,sx,nl])
y_ = tf.placeholder(tf.float32, shape=[None, out_val])


#densely-connected layer
W_fc1 = weight_variable([sy * sx * nl, 1024])
b_fc1 = bias_variable([1024])

h_pool_flat = tf.reshape(x_image, [-1, sy * sx * nl])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#readout layer

W_fc2 = weight_variable([1024, out_val])
b_fc2 = bias_variable([out_val])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


#for testing
y_conv_max = tf.argmax(y_conv,1)
y_max = tf.argmax(y_,1)

#cross-entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

#learning rate = 0.01
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#initialize the variables
init = tf.initialize_all_variables()

#save and restore variables
saver = tf.train.Saver()


#launch Session
sess = tf.InteractiveSession()
sess.run(init)

#input data here, read training data
print("({} simulation)".format(net))

if switch == 1:
    saver.restore(sess, "Weights/weights_{}.ckpt".format(net))


error_list = []

#input data here, read training data
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(super_maxiter):
    batch_xs, batch_ys = get_batch(train_data,train_batch)
    if i%super_moditer == 0:
        train_accuracy = accuracy.eval(feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob: 1.0})
        error_list.append(train_accuracy)
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x_image: batch_xs, y_: batch_ys, keep_prob: 0.5})

error_file = open("Errors/error_{}.txt".format(net), 'w')

for error in error_list:
    error_file .write("{}\n".format(error))
error_file.close

#cross-validation
print("start testing")


test_data = get_data_super(testing,classes,sy)

x_test, y_test = get_batch_grouped(test_data,class_batch,out_val)

y_predict = y_conv_max.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0})
# y_actual = y_max.eval(feed_dict={y_: y_test})

#conf_matrix

print conf_matrix(y_predict,classes,class_batch)
# print("test accuracy %g"%accuracy.eval(feed_dict={x_image: x_test, y_: y_test, keep_prob: 1.0}))

saver.save(sess, "Weights/weights_{}.ckpt".format(net))