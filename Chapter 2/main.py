'''
Title: Chapter 2
Author: Ben Brixton
'''

import numpy as np

if __name__ == "__main__":
    
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    outputs = np.dot(weights, inputs) + bias
    
    print(outputs)