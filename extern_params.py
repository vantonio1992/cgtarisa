#manual input
#nx = ny = sx = sy = 64


switch = 0
sx = sy = 64
fs1 = fs2 = 5
nf1 = 12
nf2 = 24
nl = 3
train_f = 6
train_batch = 40
test_batch = 192

image_reco = 6
test_f = 1
maxiter = 150
moditer = 25
input_size = sy*sx*nl
risa_pool = 4

learning_rate = 0.01
lambda_r = np.array([1e-4])
out_val = 3


training = "Training"

testing = "Testing"

images = ["PI", "PP", "TRU"]

encode_type = ["CNN_ae", "CNN_ae_bias", "RISA", "CNN_Hybrid"]
