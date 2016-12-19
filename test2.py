#standard packages

from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *

RISA_W1 = open('Sample/RISA_W1.pkl', 'rb')
RISA_b1 = open('Sample/RISA_b1.pkl', 'rb')
W_risa1 = tf.constant(pickle.load(RISA_W1))
b_risa1 = tf.constant(pickle.load(RISA_b1))

with tf.Session() as sess:
	print np.shape(W_risa1.eval())