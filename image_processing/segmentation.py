import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import matplotlib.image as mpimg
from skimage import color
from sys import argv

script,file=argv

img=mpimg.imread(file)
image = color.rgb2gray(img)
thresh = threshold_otsu(image)
binary = image > thresh

fig, axes = plt.subplots(ncols=4, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1, adjustable='box-forced')
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')
ax[3] = plt.subplot(1, 4, 4, adjustable='box-forced')

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('B/W Original')
ax[0].axis('off')

ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

ax[3].imshow(img)
ax[3].set_title('Color Original')
ax[3].axis('off')

plt.show()