import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

# First, load the image again
filename = "MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)

image = tf.placeholder("uint8", [None, None, 3])
slice = tf.slice(image, [1000, 0, 0], [3000, -1, -1]) # -1 means all values in given dimension, not last -1 elem

with tf.Session() as session:
    result = session.run(slice, feed_dict={image: raw_image_data})
    print(result.shape)

plt.imshow(result)
plt.show()