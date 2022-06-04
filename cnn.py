from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
import numpy as np


train_data_gen = ImageDataGenerator(rescale=1/255, rotation_range=10, width_shift_range=0, height_shift_range=0, shear_range=0.1, zoom_range=0, fill_mode='nearest')
train_generator = train_data_gen.flow_from_directory(r"D:\ML\Mini Project CNN\Clg Images", color_mode="rgb", target_size=(100, 100))

validation_data_gen = ImageDataGenerator(rescale=1/255, rotation_range=10, width_shift_range=0, height_shift_range=0, shear_range=0.1, zoom_range=0, fill_mode='nearest')
validation_generator = validation_data_gen.flow_from_directory(r"D:\ML\Mini Project CNN\Images\validation", color_mode="rgb", target_size=(100, 100))


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()

history = model.fit(train_generator, epochs=5, validation_data=validation_generator, verbose=1)
model.evaluate(validation_generator, verbose=1)

model.save("secondModel(on clg images).h5")
