import numpy as np
import tensorflow as tf

N = 1000
batch_size = 10

# x and y are placeholders for our training data
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])

w_ = 2.0
b_ = 3.0

# training dataset:
x_train_large = np.random.random((N, 1)).astype(np.float32)
y_train_large = np.multiply(x_train_large, w_) + b_

w = tf.Variable(0.1, name="W")
b = tf.Variable(1.1, name="b")

# Our model of y = W*x + b
y_model = tf.mul(x, w) + b

# Our error is defined as the square of the differences
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.initialize_all_variables()
errors = []
with tf.Session() as session:
    session.run(model)
    # Fit the line.
    n_batch = N // batch_size + (N % batch_size != 0)
    for step in range(100):
        i_batch = (step % n_batch) * batch_size
        batch = x_train_large[i_batch:i_batch + batch_size], y_train_large[i_batch:i_batch + batch_size]
        feed_dict = {x: batch[0], y: batch[1]}
        session.run(train_op, feed_dict=feed_dict)
        _, error_value = session.run([train_op, error], feed_dict=feed_dict)
        errors.append(error_value)
        if step % 200 == 0:
            print(step, error_value)
    w_value = session.run(w)
    b_value = session.run(b)
    print("Predicted model: {a}*x + {b}".format(a=w_value, b=b_value))

import matplotlib.pyplot as plt

plt.plot([np.mean(errors[i - 50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")
