import keras
import numpy as np
import cv2
from get_di import get_di

# Path: 
raw_video = np.random.rand(8, 64, 224, 224, 3)
dynamic_image = np.zeros(8, 224, 224, 3)

for i in range(8):
    dynamic_image[i] = get_di(raw_video[i])
    # cv2.imwrite('test'+str(i)+'.png', di)

 
    
