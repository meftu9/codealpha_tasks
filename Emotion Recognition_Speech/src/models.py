import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Dense,
    GlobalAveragePooling2D,
    Bidirectional,
    LSTM,
    TimeDistributed,
    Flatten,
    Reshape
)

def build_cnn_model(input_shape=(40, 228, 1), num_classes=8):
    """
    2D Convolutional Neural Network for Spectrogram / MFCC representation.
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),

        Dense(64, activation="relu"),
        Dropout(0.4),

        Dense(num_classes, activation="softmax", name="emotion_output")
    ], name="Speech_Emotion_CNN")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_lstm_model(input_shape=(228, 40), num_classes=8):
    """
    Bidirectional LSTM Network for Temporal Sequence Modeling.
    Input expects (time_steps, features) -> (228, 40).
    """
    model = Sequential([
        Input(shape=input_shape),

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.4),

        Dense(num_classes, activation="softmax", name="emotion_output")
    ], name="Speech_Emotion_BiLSTM")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_crnn_model(input_shape=(40, 228, 1), num_classes=8):
    """
    Hybrid Convolutional Recurrent Neural Network (CRNN).
    Combines Conv2D for frequency-time representation with Bi-LSTM for sequence modeling.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # shape: (20, 114, 32)

    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # shape: (10, 57, 64)

    # Permute and reshape to (time_steps, features) for LSTM
    # (batch, freq, time, channels) -> (batch, time, freq * channels)
    x = tf.keras.layers.Permute((2, 1, 3))(x)  # shape: (57, 10, 64)
    x = Reshape((57, 10 * 64))(x)              # shape: (57, 640)

    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation="softmax", name="emotion_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Speech_Emotion_CRNN")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

