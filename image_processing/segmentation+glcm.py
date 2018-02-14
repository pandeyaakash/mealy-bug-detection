import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import matplotlib.image as mpimg
from skimage import color
from skimage.feature import greycomatrix, greycoprops
from sys import argv

script,file=argv

PATCH_SIZE = 21

#implementation of segmentation
img=mpimg.imread(file)
image1 = color.rgb2gray(img)
thresh = threshold_otsu(image1)
binary = image > thresh


#implementation of glcm
image = thresh
  # select some patches from unaffected areas of the image
leaf_locations = [(474, 291), (440, 433), (466, 18), (462, 236)]
leaf_patches = []
for loc in leaf_locations:
    leaf_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

  # select some patches from affected areas of the image
bug_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
bug_patches = []
for loc in bug_locations:
    bug_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

  # compute some GLCM properties each patch
xs = []
ys = []
for patch in (leaf_patches + bug_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])



fig, axes = plt.subplots(ncols=4, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1, adjustable='box-forced')
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')
ax[3] = plt.subplot(1, 4, 4, adjustable='box-forced')

ax[0].imshow(image1, cmap=plt.cm.gray)
ax[0].set_title('B/W Original')
ax[0].axis('off')

ax[1].hist(image1.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')

ax[3].imshow(img)
ax[3].set_title('Color Original')
ax[3].axis('off')

plt.show()
