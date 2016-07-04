import tensorflow as tf
import numpy as np

# Define dimensions
d = 10     # Size of the parameter space
N = 1000   # Number of data sample

# create random data
w = .5*np.ones(d)
x_data = np.random.random((N, d)).astype(np.float32)
y_data = x_data.dot(w).reshape((-1, 1))

# Define placeholders to feed mini_batches
X = tf.placeholder(tf.float32, shape=[None, d], name='X')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# Find values for W that compute y_data = <x, W>
W = tf.Variable(tf.random_uniform([d, 1], -1.0, 1.0))
y = tf.matmul(X, W, name='y_pred')

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Before starting, initialize the variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
mini_batch_size = 100
n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
for step in range(2001):
    i_batch = (step % n_batch)*mini_batch_size
    batch = x_data[i_batch:i_batch+mini_batch_size], y_data[i_batch:i_batch+mini_batch_size]
    sess.run(train, feed_dict={X: batch[0], y_: batch[1]})
    if step % 200 == 0:
        print(step, sess.run(W))