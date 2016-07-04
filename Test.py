import tensorflow as tf
import numpy as np
import time

data = np.random.randint(1000, size=10000)

x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    t0 = time.time()
    for i in range(5):
        session.run(model)
        x = x + 1
        print(session.run(x))

    t1 = time.time()-t0
    print(t1)