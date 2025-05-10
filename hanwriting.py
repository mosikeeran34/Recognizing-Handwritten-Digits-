import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: Normalize and reshape for the neural network
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))  # Reshape to 28x28 with 1 color channel
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))  # Reshape similarly
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize the data

# Build the model (Convolutional Neural Network)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes (digits 0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Predict on a test image
prediction = model.predict(test_images[0:1])  # Predicting the first test image
predicted_label = np.argmax(prediction)
print(f"Predicted label: {predicted_label}")
print(f"True label: {test_labels[0]}")

# Display the test image and prediction
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}, True: {test_labels[0]}")
plt.show()
