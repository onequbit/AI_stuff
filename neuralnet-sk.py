#!/usr/bin/env python3
# pylint: disable=consider-using-enumerate
from random import random
import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputClassifier

from icecream import ic


def create_sklearn_integer_to_byte_model():
    """Creates an sklearn MLPRegressor model for 32-bit integer to 8-bit byte conversion."""
    estimator = MLPClassifier(
        hidden_layer_sizes=(32, 16, 8),  # Example hidden layer sizes
        activation="identity",
        solver="adam",
        max_iter=500,  # Adjust as needed
        random_state=int(random() * 4294967295),
        early_stopping=False,
        # verbose=True
    )
    model = MultiOutputClassifier(estimator)
    return estimator


def get_bit(number, bit):
    return 1 if number & bit else 0

def to_binary(number, size):
    return [get_bit(number, 2**power) for power in range(size)]

def from_bits(array):
    return sum([bit*(2**power) for power, bit in enumerate(array)])

def train(model, training_data):
    # input = [to_binary(number, 32) for number in range(len(training_data))]
    input_data = [[number] for number in range(len(training_data))]
    # output = [to_binary(number, 8) for number in training_data]
    output_data = [number for number in training_data]
    for index in range(len(training_data)):
        print(f"{[input_data[index]]=}, {[output_data[index]]=}")
        model.fit([input_data[index]],[output_data[index]])
    return model

def get_output(model, size):
    results = []
    input = [to_binary(number, 32) for number in range(size)]
    for index in input:
        prediction = model.predict(input)
        print("#"*32)
        print(f"{index=}, {prediction[index]=}")
        results.append(prediction)
    return results


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    model = joblib.load(filename)
    print(f"loaded: {model} from {filename}")
    return model


def main():
    
    filename = "neuralnet1-sk.model"
    source_data = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""
    target = [int(b) for b in source_data.encode("utf-8")] 
    if os.path.exists(filename):
        model = load_model(filename)
    else:
        model = create_sklearn_integer_to_byte_model()
        model = train(model, target)
        # save_model(model, filename)
    
    learned_output = get_output(model, len(target))
    print(f"{learned_output[0:32]=}")
    # output = [from_bits(num) for num in learned_output]
    # print(f"{output=}")
    

if __name__ == "__main__":
    main()
