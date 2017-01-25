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

train_data = get_data_super(testing,classes,sy)

print np.shape(train_data)