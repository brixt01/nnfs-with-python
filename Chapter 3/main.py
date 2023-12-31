'''
Title: Chapter 2
Author: Ben Brixton
'''

import numpy as np

if __name__ == "__main__":
    
    inputs = [
        [1.0, 2.0, 3.0, 2.5],       # Sample 1
        [2.0, 5.0, -1.0, 2.0],       # Sample 2
        [-1.5, 2.7, 3.3, -0.8],       # Sample 3
    ]

    weights = [
        [0.2, 0.8, -0.5, 1.0],      # Neuron 1
        [0.5, -0.91, 0.26, -0.5],       # Neuron 2
        [-0.26, -0.27, 0.17, 0.87]      # Neuron 3
    ]

    biases = [
        2.0,        # Neuron 1
        3.0,        # Neuron 2
        0.5     # Neuron 3
    ]

    outputs = np.dot(inputs, np.array(weights).T) + biases
    
    print(outputs)