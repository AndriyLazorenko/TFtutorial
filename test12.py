import numpy as np
import tensorflow as tf

N = 1000
mini_batch_size = 500
# x and y are placeholders for our training data
X = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
# w = tf.Variable([1.0, 2.0], name="w")
x_train = np.random.random((N, 1)).astype(np.float32)
y_train = np.multiply(x_train, 2) + 6
a = tf.Variable(1.0, name="a")
b = tf.Variable(2.0, name="b")
# Our model of y = a*x + b
# y_model = tf.mul(x, w[0]) + w[1]
y_model = tf.mul(X, a) + b

# mini_batch_siz


# Our error is defined as the square of the differences
# y = tf.matmul(X, a, name='y_pred')
error = tf.square(tf.sub(y_, y_model))
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.initialize_all_variables()
errors = []
with tf.Session() as session:
    session.run(model)
    n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
    # for i in range(1000):
    for i in range(2000):
        # print(session.run(x_train))
        i_batch = (i % n_batch) * mini_batch_size
        batch = x_train[i_batch:i_batch + mini_batch_size], y_train[i_batch:i_batch + mini_batch_size]
        # x_value, y_value = session.run([x_train, y_train])
        # session.run(train, feed_dict={X: batch[0], y_: batch[1]})
        # print(len(x_value), len(y_value))
        # dictio = {}
        # for i in range(50):
        #     dictio[x_value[i]]=y_value[i]
        feed_dict = {X: batch[0], y_: batch[1]}
        _, error_value = session.run([train_op, error], feed_dict=feed_dict)
        # print(len(error_value))
        errors.append(error_value)
        if i % 200 == 0:
            print(i, error_value)

    # w_value = session.run(w)
    a_value = session.run(a)
    b_value = session.run(b)
    # print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
    print("Predicted model: {a}x + {b}".format(a=a_value, b=b_value))

import matplotlib.pyplot as plt
plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")

    # x_values = []
    # y_values = []
    # for i in range(1000):
    #     x_values.append(np.random.rand())
    #
    # y_values= tf.add(tf.mult(x_values,2),6)
    # session.run(train_op, feed_dict={x: x_values, y: y_values})




    # w_value = session.run(w)
    # print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))