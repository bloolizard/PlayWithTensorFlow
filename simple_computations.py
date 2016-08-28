#!/usr/bin/env python

import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)

c = a + b
d = a * b

V1 = tf.constant([1.,2.])
V2 = tf.constant([3.,4.])
M = tf.constant([[1.,2.]]) # Matrix, 2d
N = tf.constant([[1.,2.],[3.,4.]]) # Matrix, 2d
K = tf.constant([[[1.,2.],[3.,4.]]]) #Tensor, 3d+

# You can also compute on tensors
V3 = V1 + V2


