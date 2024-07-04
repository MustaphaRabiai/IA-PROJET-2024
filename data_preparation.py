import numpy as np
import tensorflow as tf

def load_and_prepare_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    vehicle_classes = [1, 9]  # 1: automobile, 9: truck
    train_filter = np.isin(y_train, vehicle_classes).flatten()
    test_filter = np.isin(y_test, vehicle_classes).flatten()
    X_train, y_train = X_train[train_filter], y_train[train_filter]
    X_test, y_test = X_test[test_filter], y_test[test_filter]
    y_train = np.where(y_train == 1, 0, 1)  # 0: automobile, 1: truck
    y_test = np.where(y_test == 1, 0, 1)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, y_train, X_test, y_test
