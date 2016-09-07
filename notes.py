#cd OneDrive/NAIST/Research/Dissertation/Codes_New

import numpy as np
import timeit
import random
import decimal
import timeit
import cv2
import functions, os
from matplotlib import pyplot as plt
from functions import *

#img = cv2.imread('Images/sample0.jpeg',0)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
#plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,2,2),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Image with Sobel (x)'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

#plt.show()

#cv2.imwrite("sample_sobelx.jpeg", sobelx)


#a = np.array([1,0,0])
#x_tensor = np.array([a][a])

#print x_tensor

#a = {"first": [[0,1,2],[3,4],[5,6]], "second": [[7,8],[9,10],[11,12], [13,14]]}



#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)


#img = cv2.imread('sample_sobelx.jpeg',cv2.IMREAD_GRAYSCALE)

#cv2.imwrite("sample_sobelx_GS.jpeg", img)


#test_file = open("Bayes/test.txt", 'w')
#a = np.array([2,3,4,5])

#for a in a:
#	test_file.write(str(a)+"\n")

#test_file.close
#exec(open('functions.py').read())
#test_file = open("test_file.txt", 'r')

#username = os.environ['USER']

#print "hello_{}".format(username)

#list_dir_src = ("TCGA-05-4384-01A-01-BS1_files/15/",
#                "TCGA-38-4631-11A-01-BS1_files/15/",
#                "TCGA-05-4425-01A-01-BS1_files/15/")

#for dir_src in enumerate(list_dir_src):
#	print(dir_src)
#training = 'Training/'
#file = 'sr(0,0).jpeg'
#if os.path.isfile(os.path.join(training,"train0/",file)):
#	print os.path.join(training,"train0/",file)
#else:
#	print "N"



print timestamp()

#This works


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [3,1])

y = tf.placeholder(tf.float32)

sample_sum = tf.add(x,y)

for i in range(3):
	if i == 0:
		result = np.array([[0],[0],[0]])
		#print result
	else:
		result = sample_sum.eval(session = sess, feed_dict = {x:result, y:[[i],[i+1],[i+2]]})

print result


