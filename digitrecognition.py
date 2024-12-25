import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape the data to include a channel dimension (for CNN)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Split training data into training and validation sets (80/20 split)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # added a dropout to fix the overfitting problem
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with training and validation data
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10) # less than 10 epochs had less accuracy

# Save the trained model
model.save('handwritten_cnn.keras')

# Plot accuracy and loss graphs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# Test model on custom images
image_number = 0
while os.path.isfile(f"C:/Users/Nisa/Desktop/Digit Recognition/digits{image_number}.png"):
    try:
        img = cv2.imread(f"C:/Users/Nisa/Desktop/Digit Recognition/digits{image_number}.png", cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Image not loaded: digits{image_number}.png")
            break

        # for resizing the images just in case, it gives an error without this
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        img = cv2.bitwise_not(img)  # inverting the colors
        img = img / 255.0  # normalizing the image

        # reshaping images to match model input shape
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        predictedDigit = np.argmax(prediction)
        print(f"This digit is a {predictedDigit}")

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predictedDigit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("Error:", e)
    finally:
        image_number += 1