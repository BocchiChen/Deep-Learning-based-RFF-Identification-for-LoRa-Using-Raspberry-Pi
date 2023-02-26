# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

# Convert .h5 file into .tflite file 
model = keras.models.load_model(r'./rffi.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimize the model by reducing the size
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
with open('rffi_model.tflite', 'wb') as f:
  f.write(tflite_model)