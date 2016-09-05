import tensorflow as tf
import math
import numpy as np

sess = tf.InteractiveSession()

# Make some fake data, 1 data points
image = np.random.randint(10, size=[1,10,10]) + np.eye(10)*10

# TensorFlow placeholder
# None is for batch processing
# (-1 keeps same size)
# 10x10 is the shape
# 1 is the number of "channels"
# (like RGB color or gray)

x = tf.placeholder("float", [None, 10, 10])
x_im = tf.reshape(x, [-1,10,10,1])

### Convolutional Layer
winx = 3
winy = 3

# How many features to compute on the window

num_filters = 2

# Weight shape should match window size
# The '1' represents the number of
# input "channels" (colors)

W1 = tf.Variable(tf.truncated_normal(
    [winx, winy, 1, num_filters],
    stddev=1./math.sqrt(winx*winy)))
b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]))

# 3x3 convolution, Pad with zeros on edges

xw = tf.nn.conv2d(x_im, W1,
                  strides=[1,1,1,1],
                  padding='SAME')
h1 = tf.nn.relu(xw + b1)

# Remember to initialize!

sess.run(tf.initiaize_all_variables())

# Peek inside
H = h1.eval(feed_dict = {x: image })

# Let's take a look

import matplotlib.pyplot as plt

plt.ion()

# Original
plt.matshow(image[0])
plt.colorbar()

# Conv channel 1
plt.matshow(H[0,:,:,0])
plt.colorbar()

# Conv channel 2
plt.matshow(H[0,:,:,1])
plt.colorbar()

