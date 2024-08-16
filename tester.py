import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0


# Model accuracy testing

# model = tf.keras.models.load_model('handwritten.h5')
# loss , acc = model.evaluate(x_test,y_test)
# print(loss)
# print(acc)


#prediction with image

# # Load the image correctly
# img = cv2.imread('8.png')[:,:,0]
# img = cv2.resize(img, (28, 28))
# # Invert the image (since it's a common preprocessing step for handwritten digit recognition)
# img = np.invert(np.array([img]))
# img = img/255.0
# # img = np.array([img])
# # Load the mode
# model = tf.keras.models.load_model('handwritten.h5')
# # Make predictions
# p = model.predict(img)
# p = np.argmax(p)
# plt.imshow(img[0], cmap='gray')
# plt.title(f"Predicted Digit: {p}")
# plt.show()

# print(p)
