# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:27:20 2019

@author: Martin Ho
"""

''' Collect and Process the Data '''
import cv2
import os
import numpy as np


satisfied_dir = 'D:/Martin Ho/SUTD/DATE 2019/Data/satisfied'
unsatisfied_dir = 'D:/Martin Ho/SUTD/DATE 2019/Data/unsatisfied'
satisfied_sets = os.listdir(satisfied_dir)
unsatisfied_sets = os.listdir(unsatisfied_dir)
rectangle_dim = [40,40,40,40] # x, y, w, h

# Create a list of directories and their corresponding labels
# satisfied: [1, 0] | unsatisfied: [0, 1]
all_data_dir = []
for img_set in satisfied_sets:
    path = satisfied_dir + '/' + img_set
    all_data_dir.append([path, [1, 0]])
for img_set in unsatisfied_sets:
    path = unsatisfied_dir + '/' + img_set
    all_data_dir.append([path, [0, 1]])
    
#print(all_data_dir)
    
# number of images per dataset
#def average_images(all_data):
#    total_images = 0
#    for path in all_data:
#        image_set = os.listdir(path[0])
#        num_images = len(image_set)
#        total_images += num_images
#        print(num_images)
#        
#    average = int(total_images/len(all_data_dir))
#    return average


HM_IMAGES = 3000

# foreground extraction code
def foreground_extraction(image, rect):

    img = image
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (rect[0],rect[1],rect[2],rect[3])
    # x, y, w, h

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    img= img*mask2[:,:,np.newaxis]
    
    return img


def process_data(img_set_dir, IMG_SIZE=100):
    subject = [cv2.imread(img_set_dir + '/' + file) for file in os.listdir(img_set_dir)]
    slicing = (len(subject) - HM_IMAGES)//2
    subject = subject[slicing:-slicing]
    
    # convert images to grayscale
    colour_subject = [cv2.cvtColor(np.array(each_image), cv2.COLOR_RGBA2BGR) for each_image in subject]
    
    # resized image set
    resized_subject = [cv2.resize(np.array(each_image), (IMG_SIZE, IMG_SIZE)) for each_image in colour_subject]
    
    # foreground extracted image set
    extracted_subject = [foreground_extraction(resize_img, rectangle_dim) for resize_img in resized_subject]
    
    # graysclaed image set
    gray_subject = [cv2.cvtColor(np.array(each_image), cv2.COLOR_BGR2GRAY) for each_image in extracted_subject]
    
    return gray_subject


all_data = []


for i, dataset in enumerate(all_data_dir):
    new_set = []
    path = dataset[0]
    new_set.append(process_data(path))
    new_set.append(dataset[1])
    all_data.append(new_set)
    print('{} out of 4 completed'.format(i+1))

np.save('D:/Martin Ho/SUTD/DATE 2019/Data/data_train-{}-{}-{}.npy'.format('gray', 100, 'fge-(40,40, 40, 40)'), all_data)

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/Martin Ho/SUTD/DATE 2019/Data/satisfied/2019.6.17-9.52.57-262/2019.6.17-9.55.20-599.bmp'
IMG_SIZE = 100
rectangle_dim = [20,40,60,59]

def foreground_extraction(image, rect):

    img = image
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (rect[0],rect[1],rect[2],rect[3])
    
    # x, y, w, h

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    img= img*mask2[:,:,np.newaxis]
    
#    cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)
    
    return img

img = cv2.imread(path)
print(img)
 # convert images to grayscale
colour_subject = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
#    # resized image set
resized_subject = cv2.resize(np.array(colour_subject), (IMG_SIZE, IMG_SIZE))
#    
#    # foreground extracted image set
extracted_subject = foreground_extraction(resized_subject, rectangle_dim)
#    
#    # graysclaed image set
#gray_subject = [cv2.cvtColor(np.array(each_image), cv2.COLOR_BGR2GRAY) for each_image in extracted_subject]

#cv2.imshow('title', extracted_subject)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(extracted_subject)
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    