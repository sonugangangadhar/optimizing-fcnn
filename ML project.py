#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2DTranspose, Conv2D, Activation, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import legacy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Constants for image dimensions, batch size, and epochs
IMG_HEIGHT, IMG_WIDTH = 227, 227
BATCH_SIZE = 16  # Updated batch size
EPOCHS = 10  # Updated epochs

# Directory paths for the training, validation, and test data
base_dir = 'C:\Users\sonug\Downloads\Concrete Crack Images for Classification.rar'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Data generators for training, validation, and test datasets with rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')
validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='binary')


def create_model(base_model):
    base_model.trainable = False  # Freeze the base model layers
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))  # Define input layer
    x = base_model(inputs, training=False)  # Pass input through the base model
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Add global average pooling layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Add a dense layer for output
    model = Model(inputs, outputs)  # Create the model
    model.compile(optimizer=legacy.Adam(), loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model
    return model

# Create models using VGG16 as base models
vgg16_model = create_model(VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

for model in [vgg16_model]:
    model.compile(optimizer=legacy.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)


# Function to create the FCN model with dilated convolutions
def create_fcn_vgg16_dilated():
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    base_model.trainable = False

    # Use outputs from the last block of VGG16 before max pooling
    x = base_model.get_layer('block5_conv3').output

    # Add dilated convolution layers
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2DTranspose(512, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    x = Activation('sigmoid')(x)  # Apply sigmoid activation separately

    model = Model(inputs=base_model.input, outputs=x)
    return model

# Create, compile, and summarize the dilated FCN model
fcn_dilated_model = create_fcn_vgg16_dilated()
adam_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=5e-4)  # Use legacy Adam optimizer
fcn_dilated_model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
fcn_dilated_model.summary()

# Train the model
history = fcn_dilated_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Evaluate the model on the test data
test_loss, test_accuracy = fcn_dilated_model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()


# Define the directory for validation data for crack detection
validation_crack_dir = os.path.join(base_dir, 'validation/Positive')

# Initialize data generator for validation data
validation_crack_datagen = ImageDataGenerator(rescale=1./255)

# Create data generator for validation data
validation_crack_generator = validation_crack_datagen.flow_from_directory(
    validation_crack_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,  # No class labels for validation
    shuffle=False) # Do not shuffle the data

val_crack_predictions = fcn_dilated_model.predict(validation_crack_generator)


# Displaying a few images with their segmentation results
test_images = next(test_generator)
predicted_masks = fcn_dilated_model.predict(test_images)
num_images_to_show = 5

for i in range(num_images_to_show):
    plt.figure(figsize=(10, 4))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i])
    plt.title(f"Original Image {i}")
    plt.axis('off')

    # Display segmented image
    plt.subplot(1, 2, 2)
    segmented_image = np.where(predicted_masks[i] > 0.5, 1, 0)  # Threshold segmentation mask
    plt.imshow(segmented_image, cmap='gray')
    plt.title(f"Segmented Image {i}")
    plt.axis('off')

    plt.show()

