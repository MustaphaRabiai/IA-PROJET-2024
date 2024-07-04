import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from data_preparation import load_and_prepare_data
from model_training import train_model
from plotting import plot_training_history

# Charger le modèle entraîné
model_path = 'model/vehicle_classifier.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
else:
    model = tf.keras.models.load_model(model_path)

    def detect_vehicle(image_path):
        # Lire et prétraiter l'image
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (32, 32))
        image_array = np.expand_dims(image_resized, axis=0) / 255.0
        
        # Prédire la classe de l'image
        prediction = model.predict(image_array)
        if prediction[0] > 0.5:
            return "Truck", prediction[0][0]
        else:
            return "Automobile", 1 - prediction[0][0]

    st.title("Vehicle Classification App")

    # Initialize counters
    if 'auto_count' not in st.session_state:
        st.session_state.auto_count = 0
    if 'truck_count' not in st.session_state:
        st.session_state.truck_count = 0

    # Define tabs
    tab1, tab2, tab3 = st.tabs(["Classify Vehicle", "Statistics", "Training History"])

    with tab1:
        st.header("Classify Vehicle")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            label, confidence = detect_vehicle("temp_image.jpg")
            
            st.write(f"Predicted Label: {label} with confidence {confidence:.2f}")
            
            if label == "Automobile":
                st.session_state.auto_count += 1
                st.write("This vehicle is allowed to pass on this road (Vehicle < 3 tons).")
            else:
                st.session_state.truck_count += 1
                st.write("This vehicle is not allowed to pass on this road (Vehicle > 5 tons).")

    with tab2:
        st.header("Statistics")
        
        # Display statistics
        st.write(f"Number of Automobiles: {st.session_state.auto_count}")
        st.write(f"Number of Trucks: {st.session_state.truck_count}")
        
        # Display bar chart
        counts = [st.session_state.auto_count, st.session_state.truck_count]
        labels = ['Automobiles', 'Trucks']
        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts, color=['blue', 'green'])
        plt.xlabel('Vehicle Type')
        plt.ylabel('Count')
        plt.title('Number of Vehicles Detected')
        st.pyplot(plt)

    with tab3:
        st.header("Training History")
        
        # Load and prepare data, train model, and get training history
        X_train, y_train, X_test, y_test = load_and_prepare_data()
        history = train_model(X_train, y_train, X_test, y_test)
        
        # Plot training history
        plot_training_history(history)
