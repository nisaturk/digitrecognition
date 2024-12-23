import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# normalizing the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)), 
    tf.keras.layers.Rescaling(1./255),  # normalizing the images directly in the model
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),  # batch normalization
    tf.keras.layers.Dropout(0.3),  # higher dropout to mitigate overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),  
    tf.keras.layers.Dense(10, activation='softmax')
])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_handwritten_model.keras', save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])  # Use validation split

# loading the best saved model
model = tf.keras.models.load_model('best_handwritten_model.keras')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# processing the images
image_number = 0
while os.path.isfile(f"C:/Users/Nisa/Desktop/Digit Recognition/digits{image_number}.png"):
    try:
        img = cv2.imread(f"C:/Users/Nisa/Desktop/Digit Recognition/digits{image_number}.png", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Image not loaded: digits{image_number}.png")
            break

        # resizing image to 28x28 if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        img = cv2.bitwise_not(img)  # inverting colors
        img = img / 255.0  # normalizing the image
        img = img.reshape(1, 28, 28, 1)  # reshaping image to match model input shape (28x28x1)

        # prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)  # extracting the highest probability
        print(f"Prediction: {predicted_digit}")

        # displaying the results
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predicted_digit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("Error:", e)
    finally:
        image_number += 1