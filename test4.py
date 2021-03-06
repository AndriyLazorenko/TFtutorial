import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

# First, load the image again
filename = "22.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()
