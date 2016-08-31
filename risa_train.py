#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train risa')

import os
import sys
import csv
import numpy as np
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
import tensorflow_ae_base

from tensorflow_ae_base import *
import tensorflow_util

import myutil

exec(open('extern_params.py').read())

#
# load sample data
#

ss = 64 # sample size
if(not 'qqq_trn' in locals()):
    file_input = 'qqq_trn_w{}.npy'.format(ss)
    path_data = os.path.join(dir_input,'input_w{}'.format(ss),file_input)
    qqq_trn = np.load(path_data)
    print('load input from {}'.format(path_data))






#
# save parameters
#
weight1_fin = {k:sess.run(v) for k,v in weight1.items()}
bias1_fin = {k:sess.run(v) for k,v, in bias1.items()}
myutil.saveObject(weight1_fin,'weight1.{}.pkl'.format(stamp))
myutil.saveObject(bias1_fin,'bias1.{}.pkl'.format(stamp))

myutil.timestamp()
print('stamp1 = \'{}\''.format(stamp))
