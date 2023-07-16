"""
execute: python src/03_classifiying_handwritten_numbers.py

based on:
Chollet, François (2018): Deep learning with Python. Shelter Island, NY: Manning. Page 27-30
https://github.com/fchollet/deep-learning-with-python-notebooks
"""

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

print("Start script!")

# Loading the MNIST dataset in Keras
print("Loading data")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# info prints
print(f"train_images: {train_images.shape}")
print(f"len(train_labels): {len(train_labels)}")
print(f"train_labels: {train_labels}")
print(f"test_images.shape: {test_images.shape}")
print(f"len(test_labels): {len(test_labels)}")
print(f"test_labels: {test_labels}")


# The network architecture
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])


# The compilation step
model.compile(optimizer="rmsprop",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# "Fitting" the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)


# Using the model to make predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)

# info prints
print(f"predictions[0]: {predictions[0]}")
print(f"predictions[0].argmax(): {predictions[0].argmax()}")
print(f"predictions[0][7]: {predictions[0][7]}")
print(f"test_labels[0]: {test_labels[0]}")

# optional: display the image
image = np.reshape(test_images[0], (28, 28))
plt.imshow(image, cmap=plt.cm.binary)
plt.show()

# Evaluating the model on new data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

"""
Example output:

Start script!
Loading data
train_images: (60000, 28, 28)
len(train_labels): 60000
train_labels: [5 0 4 ... 5 6 8]
test_images.shape: (10000, 28, 28)
len(test_labels): 10000
test_labels: [7 2 1 ... 4 5 6]
2023-07-16 14:05:05.054820: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/5
469/469 [==============================] - 5s 8ms/step - loss: 0.2666 - accuracy: 0.9233
Epoch 2/5
469/469 [==============================] - 4s 9ms/step - loss: 0.1084 - accuracy: 0.9681
Epoch 3/5
469/469 [==============================] - 4s 8ms/step - loss: 0.0721 - accuracy: 0.9789
Epoch 4/5
469/469 [==============================] - 4s 8ms/step - loss: 0.0513 - accuracy: 0.9844
Epoch 5/5
469/469 [==============================] - 4s 9ms/step - loss: 0.0385 - accuracy: 0.9883
1/1 [==============================] - 0s 189ms/step
predictions[0]: [3.4175710e-07 1.3089828e-09 4.0053446e-06 2.7583319e-05 9.0716039e-11
 4.1824723e-08 3.8748878e-12 9.9996591e-01 1.2840653e-07 2.1437183e-06]
predictions[0].argmax(): 7
predictions[0][7]: 0.9999659061431885
test_labels[0]: 7
313/313 [==============================] - 1s 4ms/step - loss: 0.0662 - accuracy: 0.9799
test_acc: 0.9799000024795532

"""