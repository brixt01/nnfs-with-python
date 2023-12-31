'''
Title: Chapter 4
Author: Ben Brixton
'''

import numpy as np
from nnfs import nnfs_spiral_data
import matplotlib.pyplot as plt

class Layer_Dense:
    
    # Initialise weights and biases
    def __init__(self, n_inputs, n_neurons): # 4, 3
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        

if __name__ == "__main__":

    np.random.seed(0)       # Set random seed, so all data matches book
    x, y = nnfs_spiral_data(100, 3)     # Get sample data: x=data, y=classes

    dense1 = Layer_Dense(n_inputs=2, n_neurons=3)
    dense1.forward(x)

    print(dense1.output[:5])