import functions


#manual input
#nx = ny = sx = sy = 64

nx = 4
ny = 6
sx = sy = 32
fs = 2
nf = 2
nl = 3
sl = 1

input_size = sy*sy*nl
pool_size = input_size
pools = 2
learning_rate = 0.01
iter_max = 8
iter_print = 2

output_size = pool_size/pools

training = "Training"

testing = "Testing"

images = ["PI", "PP", "TRU"]
