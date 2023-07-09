from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

"""
Chollet, François (2018): Deep learning with Python. Shelter Island, NY: Manning.
https://github.com/fchollet/deep-learning-with-python-notebooks
Page 27-

execute: python src/03_classifiying_handwritten_numbers.py
"""

# Loading the MNIST dataset in Keras
print("Loading data")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


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
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

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