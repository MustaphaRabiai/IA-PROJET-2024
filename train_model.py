import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import numpy as np
import os

# Charger les données CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Garder uniquement les classes 1 (automobiles) et 9 (camions)
vehicle_classes = [1, 9]
train_filter = np.isin(y_train, vehicle_classes).flatten()
test_filter = np.isin(y_test, vehicle_classes).flatten()

X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

# Convertir les labels en 0 (automobile) et 1 (camion)
y_train = np.where(y_train == 1, 0, 1)
y_test = np.where(y_test == 1, 0, 1)

# Normaliser les images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Définir le modèle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Une seule sortie pour la classification binaire
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Sauvegarder le modèle
os.makedirs('model', exist_ok=True)
model.save('model/vehicle_classifier.h5')
