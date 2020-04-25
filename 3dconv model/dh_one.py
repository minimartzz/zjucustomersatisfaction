from keras.layers import Input, Dense, TimeDistributed, concatenate, LSTM, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.utils import plot_model
import numpy as np


# functions


def timedistributed_vgg_block(layer_in, num_filters, num_conv):
    for _ in range(num_conv):
        layer_in = TimeDistributed(
            Conv2D(num_filters, (3, 3), padding='same',
                   activation='relu'))(layer_in)
    layer_in = TimeDistributed(MaxPooling2D(pool_size=(2, 2),
                                            strides=(2, 2)))(layer_in)
    return layer_in

# loading data
        

"""
# Main Inputs
"""
bgra_input_1 = Input(shape=(130, 130, 4),
                     dtype='float32',
                     name='bgra_input_1')
bgra_input_2 = Input(shape=(130, 130, 4),
                     dtype='float32',
                     name='bgra_input_2')
depth_input_1 = Input(shape=(130, 130, 1),
                      dtype='float32',
                      name='depth_input_1')
depth_input_2 = Input(shape=(130, 130, 1),
                      dtype='float32',
                      name='depth_input_2')
#skeletxt_input_1 = Input(shape=(250, 75),
#                         dtype='float32',
#                         name='skeletxt_input_1')
#skeletxt_input_2 = Input(shape=(250, 75),
#                         dtype='float32',
#                         name='skeletxt_input_2')
"""
# Convolution Layers for BGRA
# 5 VGG Blocks with 2 Layers each
"""
# First Convolution Layer
bgra_input_1_conv = TimeDistributed(
    Conv2D(64, (3, 3),
           input_shape=(130, 130, 4),
           activation='relu',
           padding='same'))(bgra_input_1)
bgra_input_2_conv = TimeDistributed(
    Conv2D(64, (3, 3),
           input_shape=(130, 130, 4),
           activation='relu',
           padding='same'))(bgra_input_2)

# bgra_input_1_conv = timedistributed_vgg_block(bgra_input_1_conv, 64, 1)
bgra_input_1_conv = timedistributed_vgg_block(bgra_input_1_conv, 128, 1)
bgra_input_1_conv = timedistributed_vgg_block(bgra_input_1_conv, 256, 2)
bgra_input_1_conv = timedistributed_vgg_block(bgra_input_1_conv, 512, 2)
bgra_input_1_conv = timedistributed_vgg_block(bgra_input_1_conv, 512, 2)
bgra_input_1_conv = TimeDistributed(Flatten())(bgra_input_1_conv)

# bgra_input_2_conv = timedistributed_vgg_block(bgra_input_2_conv, 64, 1)
bgra_input_2_conv = timedistributed_vgg_block(bgra_input_2_conv, 128, 1)
bgra_input_2_conv = timedistributed_vgg_block(bgra_input_2_conv, 256, 2)
bgra_input_2_conv = timedistributed_vgg_block(bgra_input_2_conv, 512, 2)
bgra_input_2_conv = timedistributed_vgg_block(bgra_input_2_conv, 512, 2)
bgra_input_2_conv = TimeDistributed(Flatten())(bgra_input_2_conv)
"""
# Convolution Layers for Depth
# 2 VGG Blocks with 2 Conv2D Layers each
"""
# First Convolution Layer
depth_input_1_conv = TimeDistributed(
    Conv2D(32, (3, 3),
           input_shape=(130, 130, 1),
           activation='relu',
           padding='same'))(depth_input_1)
depth_input_2_conv = TimeDistributed(
    Conv2D(32, (3, 3),
           input_shape=(130, 130, 1),
           activation='relu',
           padding='same'))(depth_input_2)

# depth_input_1_conv = timedistributed_vgg_block(depth_input_1_conv, 32, 1)
depth_input_1_conv = timedistributed_vgg_block(depth_input_1_conv, 64, 1)
depth_input_1_conv = timedistributed_vgg_block(depth_input_1_conv, 128, 2)
depth_input_1_conv = TimeDistributed(Flatten())(depth_input_1_conv)

# depth_input_2_conv = timedistributed_vgg_block(depth_input_2_conv, 32, 1)
depth_input_2_conv = timedistributed_vgg_block(depth_input_2_conv, 64, 1)
depth_input_2_conv = timedistributed_vgg_block(depth_input_2_conv, 128, 2)
depth_input_2_conv = TimeDistributed(Flatten())(depth_input_2_conv)

# Concatenate
conc_out = concatenate([
    bgra_input_1_conv, bgra_input_2_conv, depth_input_1_conv,
    depth_input_2_conv
    ])

# LSTM
lstm_out = LSTM(8192)(conc_out)
lstm_out = Dense(2048, activation='relu')(lstm_out)

# Output
main_output = Dense(2, activation='softmax', name='main_output')(lstm_out)

# Model Definition
model = Model(inputs=[
    bgra_input_1, depth_input_1, bgra_input_2, depth_input_2,
],
              outputs=[main_output])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())
plot_model(model,
           show_shapes=True,
           to_file='multiple_vgg_blocks_final_convlayer.png')

# model.fit(
#     {
#         'bgra_input_1': bgra_input_1_data,
#         'bgra_input_2': bgra_input_2_data,
#         'depth_input_1': depth_input_1_data,
#         'depth_input_2': depth_input_2_data,
#         'skeletxt_input_1': skeletxt_input_1_data,
#         'skeletxt_input_2': skeletxt_input_2_data,
#         'main_output': labels
#     },
#     epochs=epoch_placeholder,
#     batch_size=batch_size_placeholder)
