import numpy as np
import cv2
import os


os.system("cls")


from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
#image = invert(data.horse())


image = cv2.imread("C:\\Users\\INKOM06\\Pictures\\washhand\\hueb\\0400.jpg")
image0 = image.copy()

kernel = np.ones((3, 3), np.uint8) 
image = cv2.erode(image, kernel)  



# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image0, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(image, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('eroded', fontsize=20)

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()