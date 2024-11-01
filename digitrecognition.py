import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2), # added a dropout to fix the overfitting problem
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)  # less than 10 epochs had less accuracy
model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

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

        img = cv2.bitwise_not(img) # inverting the colors
        img = img / 255.0 # normalization

        # checking the shape of the image before reshaping
        # print(f"Original image shape: {img.shape}")     
        # reshaping to match model input shape (1, 28, 28)
        img = img.reshape(1, 28, 28)  # reshaping for the model input

        prediction = model.predict(img)

        print(f"Raw prediction: {prediction}") # raw prediction to help with debugging later
        predictedDigit = np.argmax(prediction) # highest probability
        print(f"This digit is a {predictedDigit}")

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predictedDigit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("Error:", e)
    finally:
        image_number += 1