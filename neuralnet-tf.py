#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import environ, path

# environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# environ["CUDA_VISIBLE_DEVICES"] = "-1"
# environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PAIRING_SIZE = 100

def create_integer_to_byte_model():
    """Creates a neural network model for 32-bit integer to 8-bit byte conversion."""

    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation="relu", input_shape=(1,)
            ),  # Input: 32-bit int
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(
                8, activation="sigmoid"
            ),  # Output: 8-bit byte (0-1 range)
        ]
    )

    model.compile(
        optimizer="adam", loss="mse"
    )  # Mean Squared Error loss is appropriate here.
    return model


def train_and_predict(model, input_integer):
    """Trains (if needed) and predicts an 8-bit byte from a 32-bit integer."""

    # Reshape the input for the model
    input_data = np.array([[input_integer]], dtype=np.int32)

    # Prediction.
    prediction = model.predict(input_data)
    # Scale the output to the 0-255 range and convert to integer
    byte_output = (prediction[0] * 255).astype(np.uint8)

    return byte_output


def save_model(model, filename):
    model.save_weights(filename)


def load_model(filename):
    return tf.keras.models.load_model(filename)

def train(model):
    # Generate some training data. For a more robust model, create a larger set of training data.
    # Note: due to the nature of this problem, it is very difficult to have the network learn a general function.
    # It is much more likely to simply memorize the input/output pairs.
   
    training_input = np.random.randint(0, 2**31, size=(PAIRING_SIZE, 1))
    training_output = np.random.rand(PAIRING_SIZE, 8)  # random bytes.
    model.fit(training_input, training_output, epochs=10, verbose=1)


def main():
    # Example usage:
    filename = "neuralnet1.model"

    if path.exists(filename):
        model = load_model(filename)
    else:
        model = create_integer_to_byte_model()

    # Example input integer
    input_int = 1234567890
    # Predict the byte
    predicted_byte = train_and_predict(model, input_int)

    print(f"Input Integer: {input_int}")
    print(f"Predicted Byte: {predicted_byte}")

    # example of outputting a list of bytes.
    input_int_list = [1, 2, 3, 4, 5]
    for i in input_int_list:
        print(f"input int: {i}, output byte: {train_and_predict(model, i)}")

    save_model(model, filename)

if __name__ == "__main__":
    main()

