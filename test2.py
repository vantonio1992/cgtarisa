#standard packages

from PIL import Image

import numpy as np
import os
import timeit
import pickle, random
import datetime
from functions import *
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('train1.jpeg')

print get_layered_rgb(img)
