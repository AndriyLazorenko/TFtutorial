import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

# First, load the image again
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

print(height)
print(width)
print([width])

# This is left to understand retarded syntax of arrays of 1 element

with tf.Session() as session:
    # x = tf.reverse_sequence(x, width * [height], 0, batch_dim=1)
    # x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    x = tf.reverse_sequence(x, np.ones((height,)) * width, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()