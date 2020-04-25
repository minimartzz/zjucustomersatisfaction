import os
import numpy as np
import cv2
#import keras
#from sklearn.preprocessing import minmax_scale
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.sequence import TimeseriesGenerator


class Reminder(Exception):
    pass


# Labels
# satisfied = 0, unsatisfied = 1
# satisfied = [0,1], unsatisfied = [1,0]

labels = [
    "1 2 ZZCJ201906171033379110", "1 3 ZZCJ201906171105021935",
    "1 5 ZZCJ201906171155596915", "1 7 ZZCJ201906171324162343",
    "1 8 ZZCJ201906171349092775", "1 8 ZZCJ201906171349092775",
    "1 12 ZZCJ201906171709066130", "1 13 ZZCJ201906171727542293",
    "1 15 ZZCJ201906181636061033", "1 17 ZZCJ201906181719451669",
    "1 19 ZZCJ201906181812238185", "0 1 ZZCJ201906170957494775",
    "0 4 ZZCJ201906171128456932", "0 6 ZZCJ201906171245402386",
    "0 10 ZZCJ201906171625252817", "0 11 ZZCJ201906171700024067",
    "0 14 ZZCJ201906181610235566", "0 16 ZZCJ201906181655193615",
    "0 18 ZZCJ201906181744011711", "0 20 ZZCJ201906181833474926"
]

labels = [item.split() for item in labels]
# labels = [[int(item[0]), int(item[1]), item[2]] for item in labels]
labels = [[[0, 1] if item[0] == '0' else [1, 0],
           int(item[1]), item[2]] for item in labels]
labels.sort(key=lambda x: x[1])
# print(labels)
categorical_labels = [item[0] for item in labels]
print(categorical_labels)  # 9 Satisfied, 11 Unsatisfied
print(categorical_labels.count([0, 1]))  # 9 Satisfied, 11 Unsatisfied

# Data Set

# Train - test split

# File Processing
# num_data = ['data1', 'data2']
# type_data = ['bgraimage', 'depthimage', 'skeletontxt']
# base_dir = 'D:\\'

# data_info = {'filepath': {}, 'data_length': {}}
# for nums in num_data:
#     for info_type in data_info.keys():
#         data_info[info_type][nums] = {}
#     for types in type_data:
#         data_properties = []
#         filepath = os.path.join(base_dir, nums, types)
#         for folders in os.listdir(filepath):
#             folder_filepath = os.path.join(filepath, folders)
#             temp_list = [
#                 os.path.getctime(folder_filepath),
#                 len(os.listdir(folder_filepath)), folder_filepath
#             ]
#             if types == 'skeletontxt':
#                 skeletontxt_filepath = os.path.join(
#                     folder_filepath,
#                     os.listdir(folder_filepath)[0])
#                 with open(skeletontxt_filepath, 'r') as f:
#                     temp_skeletontxt = [line.strip() for line in f.readlines()]
#                     temp_skeletontxt = [
#                         list(map(lambda x: float(x), item.split()))
#                         for item in temp_skeletontxt
#                     ]
#                     timestep_length = len(temp_skeletontxt)
#                 temp_list[1] = timestep_length
#             data_properties.append(temp_list)
#         data_properties.sort(key=lambda x: x[0])
#         num_of_files = [item[1] for item in data_properties]
#         folders_filepath = [item[2] for item in data_properties]
#         data_info['data_length'][nums][types] = num_of_files
#         data_info['filepath'][nums][types] = folders_filepath

# for info_type in data_info.keys():

#     data_info[info_type]['data1']['depthimage'].insert(
#         12, data_info[info_type]['data1']['depthimage'].pop(11))

# print(data_info)


def pick_frames(frame_list, num_frames=500):
    total_frames = len(frame_list)
    frame_skip = total_frames / num_frames
    new_frame_list = []
    i = 0
    for _ in range(500):
        new_frame_list.append(frame_list[int(np.floor(i))])
        i += frame_skip

    return new_frame_list


def form_image_sequence(img_list,
                        folder_path,
                        sequence_array,
                        img_type,
                        resize=0):
    """
    resize - parameter (0 to 1)
    """
    for img_path in img_list:
        full_img_path = os.path.join(img_path, folder_path)
        img_array = cv2.imread(full_img_path, cv2.IMREAD_UNCHANGED)
        if resize > 1:
            raise Reminder("Input Value should be below 1.")
        elif resize != 0:
            img_array = cv2.resize(img_aray, (0, 0), fx=resize, fy=resize)
        if img_type == 'rgb':
            img_array = img_array / 255.0
        sequence_array.append(img_array)
    sequence_array = np.array(sequence_array)
    return sequence_array


# test_filepath = 'D:\\data1\\bgraimage\\2019.6.17-9.52.57-262'
# sequence_array = []
# for image in os.listdir(test_filepath)[:2]:
#     img_array = cv2.imread(os.path.join(test_filepath, image),
#                            cv2.IMREAD_UNCHANGED)
#     img_array = img_array / 255.0
#     sequence_array.append(img_array)
# sequence_array = np.array(sequence_array)
