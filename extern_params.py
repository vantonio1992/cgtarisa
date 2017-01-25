#manual input
#nx = ny = sx = sy = 64
ae_maxiter = 200
ae_moditer = 25

class_batch = 50
classes = ["PI", "PP", "TRU"]
encode_type = ["CNN_ae", "CNN_RISA", "CNN_Hybrid"]
fs1 = fs2 = es = 3
image_reco = 6
lambda_r = np.array([0.01])
learning_rate = 1e-4

nl = 3
nf1 = 12
nf2 = nf1*2

out_val = len(classes)
risa_pool = 4
super_maxiter = 150
super_moditer = 25
switch = 0
sx = sy = 128
training = "Training/"
testing = "Testing/"
time = 0
train_batch = 500


files = {'TRU': 'TCGA-05-4384-01A-01-BS1_files',
		  'PI': 'TCGA-05-4425-01A-01-BS1_files',
		  'PP': 'TCGA-38-4631-11A-01-BS1_files'
		}