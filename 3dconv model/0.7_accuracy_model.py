import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

#df_angle1_bgra = np.load('datasets/satisfied-unsatisfied-5cc531dfdd75a15e8b19e240/angle_1-bgra-500-50x50.npy', allow_pickle=True)
#df_angle2_bgra = np.load('datasets/satisfied-unsatisfied-5cc531dfdd75a15e8b19e240/angle_2-bgra-500-50x50.npy', allow_pickle=True)
#df_angle1_depth = np.load('datasets/satisfied-unsatisfied-5cc531dfdd75a15e8b19e240/angle_1-depth-500-50x50.npy', allow_pickle=True)
#df_angle2_depth = np.load('datasets/satisfied-unsatisfied-5cc531dfdd75a15e8b19e240/angle_2-depth-500-50x50.npy', allow_pickle=True)

def to_model_bgra(dataset, split=4):
    new_dict = {"data": [],"label": []}
    for dataset in dataset:
        new_dict["data"].append(dataset[0])
        new_dict["label"].append(dataset[1])
    
    train_x = np.array(new_dict['data'][:-split])
    test_x = np.array(new_dict['data'][-split:])
    train_y = np.array(new_dict['label'][:-split])
    test_y = np.array(new_dict['label'][-split:])
    
    return train_x, train_y, test_x, test_y

def to_model_depth(dataset, split=4):
    new_dict = {"data": [],"label": []}
    for dataset in dataset:
        new_dict["data"].append(dataset[0])
        new_dict["label"].append(dataset[1])
    
    for num, imageset in enumerate(new_dict['data']):
        new_dict['data'][num] = np.array(imageset).reshape([500, 50, 50, 1])
    
    train_x = np.array(new_dict['data'][:-split])
    test_x = np.array(new_dict['data'][-split:])
    train_y = np.array(new_dict['label'][:-split])
    test_y = np.array(new_dict['label'][-split:])
    
    return train_x, train_y, test_x, test_y

''' Creating Input Data '''
#trX_bgra_a1, trY_bgra_a1, teX_bgra_a1, teY_bgra_a1 = to_model_bgra(df_angle1_bgra)
#trX_bgra_a2, trY_bgra_a2, teX_bgra_a2, teY_bgra_a2 = to_model_bgra(df_angle2_bgra)
#trX_depth_a1, trY_depth_a1, teX_depth_a1, teY_depth_a1 = to_model_depth(df_angle1_depth)
#trX_depth_a2, trY_depth_a2, teX_depth_a2, teY_depth_a2 = to_model_depth(df_angle1_depth)

''' BGRA Model '''

# input layer
input_layer_bgra_a1 = Input((500, 50, 50, 4), dtype='float32', name='input_layer_bgra_a1')
## Conv Layer 1
conv_layer1_bgra = Conv3D(filters=8, kernel_size=(3, 3, 4), activation='relu')(input_layer_bgra_a1)
conv_layer1_bgra = Conv3D(filters=16, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_bgra)
conv_layer1_bgra = Conv3D(filters=32, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_bgra)
pooling_layer1_bgra = MaxPool3D(pool_size=(2, 2, 4))(conv_layer1_bgra)
## Conv Layer 2
conv_layer2_bgra = Conv3D(filters=64, kernel_size=(3, 3, 4), activation='relu')(pooling_layer1_bgra)
conv_layer2_bgra = Conv3D(filters=128, kernel_size=(3, 3, 4), activation='relu')(conv_layer2_bgra)
pooling_layer2_bgra = MaxPool3D(pool_size=(2, 2, 4))(conv_layer2_bgra)
normalized_layer_bgra = BatchNormalization()(pooling_layer2_bgra)
output_layer_bgra_a1 = Flatten()(normalized_layer_bgra)

# input layer
input_layer_bgra_a2 = Input((500, 50, 50, 4), dtype='float32', name='input_layer_bgra_a2')
## Conv Layer 1
conv_layer1_bgra = Conv3D(filters=8, kernel_size=(3, 3, 4), activation='relu')(input_layer_bgra_a2)
conv_layer1_bgra = Conv3D(filters=16, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_bgra)
conv_layer1_bgra = Conv3D(filters=32, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_bgra)
pooling_layer1_bgra = MaxPool3D(pool_size=(2, 2, 4))(conv_layer1_bgra)
## Conv Layer 2
conv_layer2_bgra = Conv3D(filters=64, kernel_size=(3, 3, 4), activation='relu')(pooling_layer1_bgra)
conv_layer2_bgra = Conv3D(filters=128, kernel_size=(3, 3, 4), activation='relu')(conv_layer2_bgra)
pooling_layer2_bgra = MaxPool3D(pool_size=(2, 2, 4))(conv_layer2_bgra)
normalized_layer_bgra = BatchNormalization()(pooling_layer2_bgra)
output_layer_bgra_a2 = Flatten()(normalized_layer_bgra)

''' Depth Model '''
input_layer_depth_a1 = Input((500, 50, 50, 1), dtype='float32', name='input_layer_depth_a1')
## Conv Layer 1
conv_layer1_depth = Conv3D(filters=16, kernel_size=(3, 3, 4), activation='relu')(input_layer_depth_a1)
conv_layer1_depth = Conv3D(filters=32, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_depth)
pooling_layer1_depth = MaxPool3D(pool_size=(2, 2, 4))(conv_layer1_depth)
## Conv Layer 2
conv_layer2_depth = Conv3D(filters=64, kernel_size=(3, 3, 4), activation='relu')(pooling_layer1_depth)
conv_layer2_depth = Conv3D(filters=128, kernel_size=(3, 3, 4), activation='relu')(conv_layer2_depth)
pooling_layer2_depth = MaxPool3D(pool_size=(2, 2, 4))(conv_layer2_depth)
output_layer_depth_a1 = Flatten()(pooling_layer2_depth)

input_layer_depth_a2 = Input((500, 50, 50, 1), dtype='float32', name='input_layer_depth_a2')
## Conv Layer 1
conv_layer1_depth = Conv3D(filters=16, kernel_size=(3, 3, 4), activation='relu')(input_layer_depth_a2)
conv_layer1_depth = Conv3D(filters=32, kernel_size=(3, 3, 4), activation='relu')(conv_layer1_depth)
pooling_layer1_depth = MaxPool3D(pool_size=(2, 2, 4))(conv_layer1_depth)
## Conv Layer 2
conv_layer2_depth = Conv3D(filters=64, kernel_size=(3, 3, 4), activation='relu')(pooling_layer1_depth)
conv_layer2_depth = Conv3D(filters=128, kernel_size=(3, 3, 4), activation='relu')(conv_layer2_depth)
pooling_layer2_depth = MaxPool3D(pool_size=(2, 2, 4))(conv_layer2_depth)
output_layer_depth_a2 = Flatten()(pooling_layer2_depth)

''' Combined MLP '''
## Concatenate
combined_layer = Concatenate()([output_layer_bgra_a1, output_layer_depth_a1, output_layer_bgra_a2, output_layer_depth_a2])

## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=128, activation='relu')(combined_layer)
dense_layer1 = Dropout(0.3)(dense_layer1)
dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.3)(dense_layer2)
output_layer = Dense(units=2, activation='softmax')(dense_layer2)

model = Model(inputs=[input_layer_bgra_a1, input_layer_bgra_a2, input_layer_depth_a1, input_layer_depth_a2],
              outputs=output_layer)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print (model.summary())
plot_model(model, show_shapes=True, to_file='D:/Martin Ho/SUTD/DATE 2019/3dconv_diagram.png')


#model.fit(x={'input_layer_bgra_a1': trX_bgra_a1,
#             'input_layer_bgra_a2': trX_bgra_a2, 
#             'input_layer_depth_a1': trX_depth_a1, 
#             'input_layer_depth_a2': trX_depth_a2}, y=trY_bgra_a1, 
#            validation_data=([teX_bgra_a1, teX_bgra_a2, teX_depth_a1, teX_depth_a2], teY_bgra_a1), 
#            verbose=1, batch_size=1, epochs=15, shuffle=True)

#model.summary() # see the summary of results of the model 
#print (model.summary())

#print ("angle1bgra, dropout: 0.3, dense layer units: 128 > 32, activation functions: relu and softmax, split: 10, 15 epochs, batch_size=1")
#score = model.evaluate(test_x, test_y, verbose=0)
#print ('Test score', score[0])
#print ('Test accuracy', score[1])


#%%
import numpy as np

def to_model_depth(dataset, split=4):
    new_dict = {"data": [],"label": []}
    for dataset in dataset:
        new_dict["data"].append(dataset[0])
        new_dict["label"].append(dataset[1])
    
    for num, imageset in enumerate(new_dict['data']):
        new_dict['data'][num] = np.array(imageset).reshape([500, 50, 50, 1])
    
    train_x = np.array(new_dict['data'][:-split])
    test_x = np.array(new_dict['data'][-split:])
    train_y = np.array(new_dict['label'][:-split])
    test_y = np.array(new_dict['label'][-split:])
    
    return train_x, train_y, test_x, test_y

df = np.load('D:/Martin Ho/SUTD/DATE 2019/Data/angle_1-depth-500-50x50.npy', allow_pickle=True)

t1, t2, t3, t4 = to_model_depth(df)
print(t1[10][40][20:30])

