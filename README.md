# Tomato-leaf-disease-detection-and-prediction-using-CNN

import os
import numpy as np # type: ignore
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Function to load images and preprocess them
def load_images(image_folder, image_size=(128, 128)):
    images = []
    labels = []
    class_names = os.listdir(image_folder)
    for label in class_names:
        class_folder = os.path.join(image_folder, label)
        if os.path.isdir(class_folder):
            for file in os.listdir(class_folder):
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(class_folder, file)
                    img = Image.open(image_path).resize(image_size)
                    img_array = np.array(img) / 255.0  # Normalize the image
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels), class_names

# Load dataset from the 'tomato_leaf_data' folder
X, y, class_names = load_images('"C:\Users\Alisha\Downloads\archive.zip"')

# Encode the labels into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply data augmentation on the training set
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)


# Building the CNN Model for Disease Classification
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Output layer for classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()


# Train the model using data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_val, y_val)
)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# Save the trained model for later use
model.save('tomato_leaf_disease_model.h5')


from flask import Flask, request, jsonify # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('tomato_leaf_disease_model.h5')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from the request
    img_file = request.files['file']
    img = Image.open(img_file).resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    # Get the class label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return jsonify({'prediction': predicted_label})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
