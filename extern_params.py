import functions


#manual input
#nx = ny = sx = sy = 64

nx = 4
ny = 6
sx = sy = 32
fs1 = fs2 = 7
nf1 = 8
nf2 = 16
nl = 3
sl = 1
train_f = 6
train_batch = 40
test_batch = 192
test_f = 1

input_size = sy*sx*nl
pool_size = input_size
pools = 2
learning_rate = 0.01
iter_max = 8
iter_print = 2
out_val = 3
output_y = sy/(pools*2)
output_x = sx/(pools*2)

training = "Training"

testing = "Testing"

images = ["PI", "PP", "TRU"]
