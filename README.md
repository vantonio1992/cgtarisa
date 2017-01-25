# Lung Cancer Image Analysis

We use various types of convolutional neural networks for an analysis of lung cancer images.

## Data Gathering
Initialize all input information using 'extern_params.py'
Some custom functions are in 'functions.py'


## FAQ

LEARNING:

Dataset can be initialized by 'functions.get_data_ae(*)'

**Format**: 'CNN_*.py' files are unsupervised learning algorithms. Running them would produce a reconstruction of random input images.

'super_*.py' files are supervised learning algorithms. The output can be produced by 'functions.conf_matrix(*)'.

The weights are saved in 'Weights/*.ckpt'