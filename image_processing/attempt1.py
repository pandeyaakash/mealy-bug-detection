import numpy as np
import skimage
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import threshold_otsu
from sys import argv

script,file=argv

image=imread(file, as_grey=True)
thresh = threshold_otsu(image)
binary = image > thresh

image=skimage.img_as_ubyte(image, force_copy=False)


"""fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('on')

ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('on')

plt.show()"""

PATCH_SIZE = 21

# select some patches from grassy areas of the image
grass_locations = [(500, 300), (291, 145), (572, 224), (365, 125)]
grass_patches = []
for loc in grass_locations:
    grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
sky_locations = [(49, 71), (39, 56), (59, 333), (73, 32)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
zs = []
ps = []
qs = []
aa = []
bb = []
cc = []
dd = []
ee = []
for patch in (grass_patches):
    glcm = greycomatrix(patch, [5], [0],256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])
    zs.append(greycoprops(glcm, 'energy')[0, 0])
    ps.append(greycoprops(glcm, 'homogeneity')[0, 0])
    qs.append(greycoprops(glcm, 'contrast')[0, 0])

for patch in (sky_patches):
    glcm = greycomatrix(patch, [5], [0],256, symmetric=True, normed=True)
    aa.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    bb.append(greycoprops(glcm, 'correlation')[0, 0])
    cc.append(greycoprops(glcm, 'energy')[0, 0])
    dd.append(greycoprops(glcm, 'homogeneity')[0, 0])
    ee.append(greycoprops(glcm, 'contrast')[0, 0])



# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        label='Infected')
ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        label='Normal')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
print("Infected patch properties:")
print("Dissimilarity:")
print(xs)
print("Correlation:")
print(ys)
print("Energy:")
print(zs)
print("Homogeneity:")
print(ps)
print("Contrast:")
print(qs)

print("Normal patch properties:")
print("Dissimilarity:")
print(aa)
print("Correlation:")
print(bb)
print("Energy:")
print(cc)
print("Homogeneity:")
print(dd)
print("Contrast:")
print(ee)


