'''
Title: Chapter 3
Author: Ben Brixton
'''

import numpy as np
from nnfs import nnfs_init, nnfs_spiral_data
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nnfs_init()     # Set up nnfs module, used for data set
    x, y = nnfs_spiral_data(100, 3)     # Get sample data: x=data, y=classes

    print(x)
    print("\n")
    print(y)

    plt.scatter(x[:,0], x[:,1], c=y, cmap='brg')     # Show sample data
    plt.show()
    
    inputs = [
        [1.0, 2.0, 3.0, 2.5],       # Sample 1
        [2.0, 5.0, -1.0, 2.0],       # Sample 2
        [-1.5, 2.7, 3.3, -0.8],       # Sample 3
    ]

    # Hidden layer 1
    weights1 = [
        [0.2, 0.8, -0.5, 1.0],      # Neuron 1
        [0.5, -0.91, 0.26, -0.5],       # Neuron 2
        [-0.26, -0.27, 0.17, 0.87]      # Neuron 3
    ]
    biases1 = [
        2.0,        # Neuron 1
        3.0,        # Neuron 2
        0.5     # Neuron 3
    ]

    # Hidden layer 2
    weights2 = [
        [0.1, -0.14, 0.5],      # Neuron 1
        [-0.5, 0.12, -0.33],       # Neuron 2
        [-0.44, 0.73, -0.13]      # Neuron 3
    ]
    biases2 = [
        -1.0,        # Neuron 1
        2.0,        # Neuron 2
        -0.5     # Neuron 3
    ]

    layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

    layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
    
    print(layer2_outputs)