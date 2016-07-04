import matplotlib.image as mpimg

# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Print out its shape
print(image.shape)

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()