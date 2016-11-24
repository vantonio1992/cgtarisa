from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *


train_batch = 1
nf1 = 2
sy = 3
sx = 3

x = np.array([[[[0,1,2],[3,4,5],[6,7,8]],[[9,10,11],[12,13,14],[15,16,17]]]])

#print np.shape(x)

a = tf.placeholder(tf.float32)
b = tf.reshape(a,[-1,nf1,sy*sx,1])
c = tf.tile(b,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
d = tf.reshape(c,[-1,nf1,sy,sx*2])
e = tf.tile(d,tf.to_int32(tf.constant(np.array([1,1,1,2]))))
x_unpool = tf.reshape(e, [-1,nf1,sy*2,sx*2])

init = tf.initialize_all_variables()

#launch Session
sess = tf.InteractiveSession()
sess.run(init)


print x_unpool.eval(feed_dict = {a: x})

