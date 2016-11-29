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
sy = 4
sx = 4

def sample(train_batch,nf1,sy,sx):
	x = []
	for data in range(train_batch):
		temp0 = []
		for i in range(nf1):
			temp1 = []
			for j in range(sy):
				temp1.append(range(sy*sx*i+sx*j,sy*sx*i+sx*j+sx))
			temp0.append(temp1)
		x.append(temp0)

	return np.array(x)

x = sample(train_batch,nf1,sy,sx)

print x


print np.shape(get_sum_2x2(x,train_batch,nf1,sy,sx))

# a = tf.placeholder(tf.float32, shape = [train_batch,nf1,sy,sx])
# #split(split_dum, num_split, tens)


# init = tf.initialize_all_variables()

# #launch Session
# sess = tf.InteractiveSession()
# sess.run(init)

# print b.eval(feed_dict = {a: sample(train_batch,nf1,sy,sx)})

# print c.eval(feed_dict = {a: sample(train_batch,nf1,sy,sx)})

