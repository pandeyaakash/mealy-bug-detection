import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io; io.use_plugin('matplotlib')
from skimage import color
from sys import argv
script,file =argv

from skimage.morphology import reconstruction
# from skimage.morphology import erosion
# from skimage.morphology import disk
# Convert to float: Important for subtraction later which won't work with uint8
img=mpimg.imread(file)
image = color.gray2rgb(img)
# imag=invert(file,dtype='float')
# skeleton = skeletonize(imag)


seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image
# selem = disk(6)
dilated = reconstruction(seed, mask, method='dilation')
# erosed = erosion(seed,selem)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(8, 2.5),
                                    sharex=True,
                                    sharey=True)

ax0.imshow(image, cmap='gray')
ax0.set_title('original image')
ax0.axis('off')
ax0.set_adjustable('box-forced')

ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
ax1.set_title('dilated')
ax1.axis('off')
ax1.set_adjustable('box-forced')

ax2.imshow(image - dilated, cmap='gray')
ax2.set_title('image - dilated')
ax2.axis('off')
ax2.set_adjustable('box-forced')

# ax3.imshow(skeleton,cmap=plt.cm.gray)
# ax3.set_title('image - skeleton')
# ax3.axis('off')
# ax3.set_adjustable('box-forced')

fig.tight_layout()
plt.show()
