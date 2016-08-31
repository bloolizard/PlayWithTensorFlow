import tensorflow as tf
import numpy as np
import math
%autoindent

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# Set random seed
np.random.seed(0)

# Load data
data = np.load('data_with_labels.npz')
train = data['arr_0']/255.
labels = data['arr_1']

# Look at some data
print(train[0])
print(labels[0])

# If you have matplotlib installed
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.ion()

def to_onehot(labels, nclasses=5):
    '''
    Convert labels to "one-hot" format.
    >>> a = [0,1,2,3]
    >>> to_onehot(a,5)
    array([[1.,0.,0.,0.,0.],
        [0., 1., 0., 0., 0.],
        [0.,0., 1., 0., 0.],
        [0., 0., 0., 1., 0.]])
    '''
    outlabels = np.zeros((len(labels), nclasses))
    for i, l in enumerate(labels):
        outlabels[i,l] = 1
    return outlabels

onehot = to_onehot(labels)

# Split data into training and validation
indices = np.random.permutation(train.shape[0])
valid_cnt = int(train.shape[0] * 0.1)
test_idx, training_idx = indices[:valid_cnt], indices[valid_cnt:]
test, train = train[test_idx, :], train[training_idx, :]
onehot_test, onehot_train = onehot[test_idx, :], onehot[training_idx, :]

sess = tf.InteractiveSession()

# These will be inputs
## Input pixels, flattened
x = tf.placeholder("float", [None, 1296])

## Known labels
y_ = tf.placeholder("float", [None, 5])

# Hidden Layer 1
num_hidden1 = 128
W1 = tf.Variable(tf.truncated_normal([1296, num_hidden1], stddev=1./math.sqrt(1296)))
b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden1]))
h1 = tf.sigmoid(tf.matmul(x, W1) + b1)

# Hidden Layer 2
num_hidden2 = 32
W2 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=2./math.sqrt(num_hidden1)))


