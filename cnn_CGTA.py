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
size = 32
files = 2
source = "Training"


#weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

  
#gathering data from images
exec(open('extern_params.py').read())

img_dict = {}
for image in images:
    img_list = get_slice(training, "%s.jpeg" % (image), sx)
    img_dict[image] = img_list["subregions"]

data_set = []
for name in images:
    one_hot = np.zeros(3)
    one_hot[images.index(name)] = 1
    for row in img_dict[name]:
        for img in img_dict[name][row]:
            data_set.append((get_rgb(img).flatten(),one_hot))

data_set = np.array(data_set)
x_test, y_test = zip(*data_set)

#start of implementation
x = tf.placeholder(tf.float32, [None, input_size])

#weights, bias incorporation

W = tf.Variable(tf.zeros([input_size, 3]))
b = tf.Variable(tf.zeros([3]))

#model implementation
y = tf.nn.softmax(tf.matmul(x, W) + b)


#Training start

#cross-entropy
y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#learning rate = 0.01
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize the variables
init = tf.initialize_all_variables()

#launch Session
sess = tf.Session()
sess.run(init)


#input data here, read training data

for i in range(1000):
    batch_xs, batch_ys = get_batch(data_set,100)
    sess.run(train_step, feed_dict={x: np.array(batch_xs), y_: np.array(batch_ys)})

 #model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))