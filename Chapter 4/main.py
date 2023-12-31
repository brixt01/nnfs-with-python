'''
Title: Chapter 4
Author: Ben Brixton
'''

import numpy as np      # Used for random numbers and matrix operations
from nnfs import nnfs_spiral_data       # Used for sample data
import matplotlib.pyplot as plt     # Used for viewing sample data

# Dense layer
# Most common type of layer. All neurons in the previous layer and connected to
# all neurons in the next layer. Each connection has a weight, and each neuron
# has a bias. 
class Layer_Dense:
    
    # Initialise weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)      # Weights: Rows = each input, Cols = each neuron
        self.biases = np.zeros((1, n_neurons))      # Biases: Rows = each neuron

    # Forward pass
    def forward(self, inputs):      # Inputs: Rows = each sample, Cols = each input (x and y coords)
        self.output = np.dot(inputs, self.weights) + self.biases        # Output: Rows = each sample, Cols = each neuron

if __name__ == "__main__":

    np.random.seed(0)       # Set random seed, so all data matches book
    x, y = nnfs_spiral_data(100, 3)     # Get sample data: x=data, y=classes

    # First hidden layer, 2 inputs (x and y coords), three neurons
    dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
    dense1.forward(x)

    print(dense1.output[:5])