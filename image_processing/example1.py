from PIL import Image
import numpy as np

# import matplotlib
# matplotlib.use('QT4Agg')
import matplotlib.pyplot as plt

plt.figure(1)                # the first figure
plt.plot([1, 2, 3])
plt.plot([4, 5, 6])


            # a second figure
plt.plot([4, 8, 12])          # creates a subplot(111) by default

           # make subplot(211) in figure1 current
plt.title('This is a graph') # subplot 211 title

plt.savefig('figure.png')
img = Image.open('15     .jpg').convert('RGB')
arr = np.array(img)
imp=open('figure.png','r')
image_data=imp.read()

for char in image_data[:4]:
    print (ord(char))  #need to use ord so we can see the number

# record the original shape
shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = arr.ravel()

# convert it to a matrix
vector = np.matrix(flat_arr)

# do something to the vector
vector[:,::10] = 128

# reform a numpy array of the original shape
arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
img2 = Image.fromarray(arr, 'RGB')
# plt.show()
img2.show()
#create string from numpy array
# aa=arr.tostring()
# print (aa)
