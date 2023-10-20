import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Directory containing the binary classification dataset (tumor yes or no)
binary_image_directory = 'datasets/'

# Directory containing the multi-class classification dataset (glioma, meningioma, pituitary)
multi_image_directory = 'D:/datasets/Training/'

# Binary classification (tumor yes or no)
no_tumor_images = os.listdir(binary_image_directory + 'no/')
yes_tumor_images = os.listdir(binary_image_directory + 'yes/')
binary_dataset = []
binary_label = []

# Multi-class classification (glioma, meningioma, pituitary)
tumor_types = ['glioma', 'meningioma', 'pituitary']
multi_dataset = []
multi_label = []

INPUT_SIZE = 64

# Load images and labels for binary classification (tumor yes or no)
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(binary_image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        binary_dataset.append(np.array(image))
        binary_label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(binary_image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')

        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        binary_dataset.append(np.array(image))
        binary_label.append(1)

# Load images and labels for multi-class classification (glioma, meningioma, pituitary)
for class_id, tumor_type in enumerate(tumor_types):
    tumor_type_folder = os.path.join(multi_image_directory, tumor_type)
    tumor_type_images = os.listdir(tumor_type_folder)

    for image_name in tumor_type_images:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(tumor_type_folder, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            multi_dataset.append(np.array(image))
            multi_label.append(class_id)

# Binary classification dataset
binary_dataset = np.array(binary_dataset)
binary_label = np.array(binary_label)

# Split the binary data into training and testing sets
binary_x_train, binary_x_test, binary_y_train, binary_y_test = train_test_split(binary_dataset, binary_label, test_size=0.2, random_state=0)

# Preprocess the binary data
binary_x_train = normalize(binary_x_train, axis=1)
binary_x_test = normalize(binary_x_test, axis=1)

binary_y_train = to_categorical(binary_y_train, num_classes=2)
binary_y_test = to_categorical(binary_y_test, num_classes=2)

# Multi-class classification dataset
multi_dataset = np.array(multi_dataset)
multi_label = np.array(multi_label)

# Split the multi-class data into training and testing sets
multi_x_train, multi_x_test, multi_y_train, multi_y_test = train_test_split(multi_dataset, multi_label, test_size=0.2, random_state=0)

# Preprocess the multi-class data
multi_x_train = normalize(multi_x_train, axis=1)
multi_x_test = normalize(multi_x_test, axis=1)

multi_y_train = to_categorical(multi_y_train, num_classes=3)  # 3 classes for glioma, meningioma, pituitary
multi_y_test = to_categorical(multi_y_test, num_classes=3)

# Model for binary classification
binary_model = Sequential()

binary_model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
binary_model.add(Activation('relu'))
binary_model.add(MaxPooling2D(pool_size=(2, 2)))

binary_model.add(Flatten())
binary_model.add(Dense(64))
binary_model.add(Activation('relu'))
binary_model.add(Dropout(0.5))
binary_model.add(Dense(2))  # Number of classes for binary classification (tumor yes or no)
binary_model.add(Activation('softmax'))

# Compile the binary model
binary_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the binary model
binary_model.fit(binary_x_train, binary_y_train, batch_size=16, verbose=1, epochs=10, validation_data=(binary_x_test, binary_y_test), shuffle=False)

# Save the trained binary model
# binary_model.save('BinaryTumorModel.h5')

# Model for multi-class classification
multi_model = Sequential()

multi_model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
multi_model.add(Activation('relu'))
multi_model.add(MaxPooling2D(pool_size=(2, 2)))

multi_model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
multi_model.add(Activation('relu'))
multi_model.add(MaxPooling2D(pool_size=(2, 2)))

multi_model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
multi_model.add(Activation('relu'))
multi_model.add(MaxPooling2D(pool_size=(2, 2)))

multi_model.add(Flatten())
multi_model.add(Dense(64))
multi_model.add(Activation('relu'))
multi_model.add(Dropout(0.5))
multi_model.add(Dense(3))  # Number of tumor types (glioma, meningioma, pituitary)
multi_model.add(Activation('softmax'))

# Compile the multi-class model
multi_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the multi-class model
multi_model.fit(multi_x_train, multi_y_train, batch_size=16, verbose=1, epochs=10, validation_data=(multi_x_test, multi_y_test), shuffle=False)

# Save the trained multi-class model
multi_model.save('MultiClassTumorModel.h5')
