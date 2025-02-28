#!/usr/bin/env python3
import math
import os
from random import random
import numpy as np
# from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from icecream import ic
import joblib


SOURCE_DATA = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""


def create_and_train_memorization_model(sequence):
    """Creates and trains an MLPRegressor to memorize a specific sequence."""
    width = round(math.log2(len(sequence)))
    model = DecisionTreeRegressor(
        max_features=width,
        random_state=int(random() * 4294967295),
    )

    # Create input-output pairs
    X = np.arange(len(sequence)).reshape(-1, 1)  # Input: index of the sequence
    y = np.array(sequence)  # Output: the sequence itself
    model.fit(X, y)
    return model

def predict_sequence(model, length):
    """Predicts the memorized sequence for the given length."""
    X_predict = np.arange(length).reshape(-1, 1)
    return model.predict(X_predict)

def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    model = joblib.load(filename)
    print(f"loaded: {model} from {filename}")
    return model

def main():
    # Example usage:
    # memorized_sequence = [10, 25, 50, 75, 100, 150, 200, 12, 45, 99]
    memorized_sequence = [int(b) for b in SOURCE_DATA.encode("utf-8")]

    filename = "memorizer.model"

    if os.path.exists(filename):
        model = load_model(filename)
    else:
        model = create_and_train_memorization_model(memorized_sequence)
        # model = train(model, target)
        save_model(model, filename)   

    # Predict the memorized sequence
    predicted_sequence = predict_sequence(model, len(memorized_sequence))
    print("Memorized Sequence:", memorized_sequence)
    print("Predicted Sequence:", predicted_sequence)
    output = bytes([int(number) for number in predicted_sequence])
    ic(SOURCE_DATA)
    ic(output)

    # output = predicted_sequence
    # print(output)

    # #Test the model on longer sequences.
    # predicted_long_sequence = predict_sequence(model, len(memorized_sequence) + 3)
    # print("Predicted longer sequence:", predicted_long_sequence)

    # #Test the model on shorter sequences.
    # predicted_short_sequence = predict_sequence(model, len(memorized_sequence) - 3)
    # print("Predicted shorter sequence:", predicted_short_sequence)

if __name__ == "__main__":
    main()
