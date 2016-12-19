#manual input
#nx = ny = sx = sy = 64

training = "Training/"

testing = "Testing/"

classes = ["PI", "PP", "TRU"]
out_val = len(classes)

encode_type = ["CNN_ae", "CNN_RISA", "CNN_Hybrid"]

switch = 1
time = 0
sx = sy = 64
fs1 = fs2 = 7
nf1 = 32
nf2 = 64
nl = 3

train_batch = 100
class_batch = 50


image_reco = 6

ae_maxiter = 1000
ae_moditer = 50
super_maxiter = 150
super_moditer = 25
risa_pool = 4

learning_rate = 1e-4
lambda_r = np.array([0.01])