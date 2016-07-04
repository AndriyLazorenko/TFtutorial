import tensorflow as tf

import time

t0 = time.time()
x = tf.placeholder("float", 3)
# x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)
    t1 = time.time()-t0
    print(t1)