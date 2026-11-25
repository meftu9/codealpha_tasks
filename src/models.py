"""
Model Architecture Module for Handwritten Character Recognition.
Defines CNN layers, dropout rates, activation functions, compilation,
and saving/loading utilities.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 47,
    dropout_rate: float = 0.25,
) -> models.Sequential:
    """
    Builds a robust Convolutional Neural Network (CNN) for handwritten character recognition
    supporting both digits (0-9) and letters (A-Z, a-z).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_3"),
        layers.BatchNormalization(name="bn_2"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_4"),
        layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),

        # Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense_256"),
        layers.Dropout(0.4, name="dropout_dense"),
        layers.Dense(num_classes, activation="softmax", name="output_classes"),
    ], name="handwritten_character_cnn")

    return model


def compile_model(
    model: models.Sequential,
    learning_rate: float = 0.001,
) -> models.Sequential:
    """
    Compiles the CNN model with Adam optimizer, Sparse Categorical Crossentropy,
    and Accuracy metric.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model: models.Model, filepath: str) -> None:
    """
    Saves the trained model to disk in .keras format.
    """
    model.save(filepath)
    print(f"Model saved successfully to: {filepath}")


def load_trained_model(filepath: str) -> models.Model:
    """
    Loads a trained Keras model from disk.
    """
    model = tf.keras.models.load_model(filepath)
    print(f"Loaded trained model from: {filepath}")
    return model
