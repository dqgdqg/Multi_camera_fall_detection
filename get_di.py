import numpy as np
import cv2
import os

def get_di(img_list):
    frames_num = len(img_list)
    img = img_list[0]
    
    height = img.shape[0]
    width = img.shape[1]
    
    frames = np.zeros((frames_num, height, width, 3), dtype='float')
    Y = np.zeros((height, width, 3), dtype='float')
    
    fw= np.ones((frames_num, height, width), dtype='float')
    
    for i in range(frames_num):
        img = img_list[i]
        img = img.astype('float') / 255
        frames[i] = img
    
    if frames_num == 1:
        fw = [1]
    else:
        for i in range(frames_num):
            fw[i] = np.full((height, width), sum((2 * np.array(np.arange(i+1,frames_num+1)) - frames_num - 1) / np.array(np.arange(i+1,frames_num+1))))
    
    for i in range(frames_num):
        T = frames[i]
        T[:,:,0] *= fw[i]
        T[:,:,1] *= fw[i]
        T[:,:,2] *= fw[i]
        Y = Y + T

    di = Y
    di = di - np.min(di)
    di = 255 * (di / np.max(di))
    di = di.astype(np.uint8)
    return di