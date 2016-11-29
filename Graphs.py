import numpy as np
import os
import timeit
import pickle, random
import math
import matplotlib.pyplot as plt
#from functions import *


#gathering data from images
exec(open('extern_params.py').read())

#exec(open('CNN_RISA.py').read())

x_np = np.linspace(0,maxiter, num = maxiter/moditer, endpoint = False)

error_np = {}
for ae_type in encode_type:
	error_file = open("{}_error_{}x{}.txt".format(ae_type,sy,sx), 'r')
	error_np[ae_type] = np.array([math.log(float(x[:-1])) for x in error_file.readlines()])
	error_file.close()





# plt.figure(1)

plt.xlabel('Training Steps')
plt.ylabel('log(Error)')
plt.title("Training of {}x{} Images".format(sy,sx))
# plt.subplot(111)
cnn = plt.plot(x_np, error_np['CNN_ae'], 'r--', label = "{}".format("CNN_ae"))
risa = plt.plot(x_np, error_np['RISA'], 'b--', label = "{}".format("RISA"))
plt.legend(loc='upper right')
plt.show()