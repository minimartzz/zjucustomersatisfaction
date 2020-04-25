# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:11:57 2019

@author: Martin Ho
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#path = 'D:/Martin Ho/SUTD/DATE 2019/Data/satisfied'
path = 'E:/data1/bgraimage'
folder_dir = [os.path.join(path, folder_path) for folder_path in os.listdir(path)]
cropper = [400, 1000, 750, 1450] # for angle 1
#cropper = [300, 1080, 1300, 1920] # for angle 2
slices = 500 # how many images per dataset
img_size = 130 # size of images

# satisfied = [0,1], unsatisfied = [1,0]

#print(folder_dir)
final_set = []


def process_images(folder_path, slices=500, img_size=50):
    paths = [os.path.join(folder_path, image) for image in os.listdir(folder_path)]
#    print(paths)
    cutter = int((len(paths) - 500)//2)
    paths = paths[cutter:len(paths)-cutter]
    
    if len(paths) > slices:
        paths = paths[0:slices]
        
    if len(paths) == slices - 1:
        paths.append(paths[-1])

    new_imageset = []
#    print(len(paths))
    print(folder_path)
    print(len(paths))
    for num, image_path in enumerate(paths):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            cropped_image = image[cropper[0]:cropper[1], cropper[2]:cropper[3]]
            resized_image = cv2.resize(np.array(cropped_image), (img_size, img_size))
            new_imageset.append(resized_image)
            
        except Exception as e:
            pass
    
    return new_imageset


for index, folder_paths in enumerate(folder_dir):
    temp = []
    new_imageset = process_images(folder_paths, slices, img_size)
    temp.append(new_imageset)
    if folder_paths.split('!')[1] == '[0,1]':
        temp.append([0,1])
    else:
        temp.append([1,0])
    final_set.append(temp)
    print('folder ', index+1, ' out of ', len(folder_dir), ' completed')
    print(len(temp))
    
print(len(final_set), len(final_set[0]))
np.save('D:/Martin Ho/SUTD/DATE 2019/Data/angle_1-bgra-500-50x50.npy', final_set)