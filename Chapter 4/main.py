'''
Title: Chapter 4
Author: Ben Brixton
'''

import numpy as np      # Used for random numbers and matrix operations
from nnfs import nnfs_spiral_data       # Used for sample data
import matplotlib.pyplot as plt     # Used for viewing sample data

# Dense layer
#
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

# ReLU (Rectified Linear Unit) activation function
#
# All inputs less than zero are set to zero, otherwise they are unchanged 
# (follows rectified linear graph). This activation function is used as a
# fast and efficient way of introducting non-linearity
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)     # For each element, set to 0 if value<0

if __name__ == "__main__":

    np.random.seed(0)       # Set random seed, so all data matches book
    x, y = nnfs_spiral_data(100, 3)     # Get sample data: x=data, y=classes

    # First hidden layer, 2 inputs (x and y coords), three neurons
    dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
    activation1 = Activation_ReLU()
    
    dense1.forward(x)
    activation1.forward(dense1.output)

    print(activation1.output[:5])