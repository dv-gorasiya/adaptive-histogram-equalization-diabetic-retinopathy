#!/usr/bin/env python
# coding: utf-8

# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
from skimage.io import imread
from skimage import exposure, color
from skimage.transform import resize

import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

img = imread(PROJ_WD+'/full_train/16_left.jpeg')  # this is a PIL image
plt.imshow(img)
plt.show()


# In[41]:


img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.30)
plt.imshow(img_adapteq)
plt.show()


# In[46]:


import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte

from skimage.exposure import histogram

from skimage.color import rgb2gray

    datagen = ImageDataGenerator(
            zca_whitening=True,
            zoom_range=0.3,
            fill_mode='nearest',
            horizontal_flip=False,
            vertical_flip=True,
            preprocessing_function=AHE)

grayscale = rgb2gray(img_adapteq)

noisy_image = img_as_ubyte(grayscale)
hist, hist_centers = histogram(noisy_image)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(noisy_image, interpolation='nearest', cmap=plt.cm.gray)
ax[0].axis('off')

ax[1].plot(hist_centers, hist, lw=2)
ax[1].set_title('Histogram of grey values')

plt.tight_layout()


# In[47]:


img_eq = exposure.equalize_hist(img)

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)


# In[ ]:




