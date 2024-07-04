import tensorflow as tf
import numpy as np
import cv2
import sys

# Charger le modèle entraîné
model = tf.keras.models.load_model('model/vehicle_classifier.h5')

def detect_vehicle(image_path):
    # Lire et prétraiter l'image
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (32, 32))
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    
    # Prédire la classe de l'image
    prediction = model.predict(image_array)
    if prediction[0] > 0.5:
        return "Truck", prediction[0]
    else:
        return "Automobile", 1 - prediction[0]

# Tester la fonction de détection
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect.py <image_path>")
    else:
        image_path = sys.argv[1]
        label, confidence = detect_vehicle(image_path)
        print(f"Predicted Label: {label} with confidence {confidence:.2f}")
