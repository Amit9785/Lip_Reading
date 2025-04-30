from tensorflow.keras.layers import Conv3D, MaxPool3D, Activation, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense, Lambda
import tensorflow as tf
from tensorflow.keras.models import Sequential
from util import load_data, num_to_char, char_to_num
from tensorflow.keras.layers import Reshape
import os
from typing import List
import cv2

def load_model()->Sequential:
    """
    Load the LipNet model architecture.
    
    Returns:
        model (Sequential): The LipNet model.
    """
        # Initialize a Sequential model
    model = Sequential()

    # Add a Conv3D layer with 128 filters, kernel size of 3, and padding of 'same', with input shape of (75, 70, 180, 1)
    model.add(Conv3D(128, 3, input_shape=(75, 70, 180, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Add another Conv3D layer with 256 filters, kernel size of 3, and padding of 'same'
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Add another Conv3D layer with 75 filters, kernel size of 3, and padding of 'same'
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Optionally print the shape after Conv3D layers (for debugging)
    # model.add(Lambda(lambda x: print(x.shape)))

    # Add a TimeDistributed layer with Flatten
    model.add(TimeDistributed(Reshape((-1,))))

    # Optionally print the shape after TimeDistributed(Flatten) (for debugging)
    # model.add(Lambda(lambda x: print(x.shape)))

    # Add a Bidirectional LSTM layer with 128 units, orthogonal kernel initializer, and return_sequences=True
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Add another Bidirectional LSTM layer with 128 units, orthogonal kernel initializer, and return_sequences=True
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Use a He normal initializer and softmax activation for the final dense layer
    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
    model.load_weights(os.path.join('..','models','checkpoint.weights.h5'))
    return model