#standard packages

from PIL import Image

import numpy as np
import os
import tensorflow as tf
import timeit
import pickle, random
import datetime
from functions import *


train_batch = 2
nf1 = 3
sy = 4
sx = 4

x = sample(train_batch,sy,sx,nf1)

x_tr = np.transpose(x,(0,3,1,2))

y = get_sum_2x2(x_tr,train_batch,nf1,sy,sx)

print x_tr
print y

