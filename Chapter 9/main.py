'''
Title: Chapter 9
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
        self.inputs = inputs        # Remember inputs - needed for backpropogation

    # Backward pass
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU (Rectified Linear Unit) activation function
#
# All inputs less than zero are set to zero, otherwise they are unchanged 
# (follows rectified linear graph). This activation function is used as a
# fast and efficient way of introducting non-linearity
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)     # For each element, set to 0 if value<0

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation function
#
# Used for classification. 
# Eponenentiates all values (to get rid of negatives), then divides them each 
# By the sum of outputs for the current sample. AKA the outputs end up being 
# confidence values, with a sum of 1. Maximum is subtracted from all, to stop
# exploding values
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))     # Subtract max then exponentiate each value
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)      # Divide by the sum of given sample's outputs
        self.output = probabilities     # Output is a confidence score for each output, summing to 1

# (common) loss
#
# Used as a base for other loss calculation
class Loss:

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)     # Calculate sample losses
        data_loss = np.mean(sample_losses)      # Calculate mean loss
        return data_loss        # Return average loss

# CCE (Categorical Cross-Entropy) loss
#
# Used for calculating loss for categories, when given a softmax output and
# the target values (as either list of categorical or one-hot encoded labels)
class Loss_CCE(Loss):

    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)     # Number of samples in batch

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)      # Clip to prevent division by 0

        # If given as categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(n_samples),
                y_true
            ]

        # If given as one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        # Calculate losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self, dvalues, y_true):
        num_samples = len(dvalues)
        num_labels = len(dvalues[0])

        if len(y_true.shape) == 1:      # Convert to one-hot encoded labels
            y_true = np.eye(num_labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs/num_samples

if __name__ == "__main__":

    '''
    Initlialising
    '''

    # Get sample data (coordinates of spiral points w/ 3 classes)
    np.random.seed(0)
    x, y = nnfs_spiral_data(100, 3)

    # Input layer
    dense1 = Layer_Dense(n_inputs=2, n_neurons=3)       # 2 inputs (x, y), 3 neurons
    activation1 = Activation_ReLU()     # ReLU activation function
    
    # Output layer
    dense2 = Layer_Dense(3, 3)      # 3 inputs, 3 neurons
    activation2 = Activation_Softmax()      # Softmax activation function

    # Loss Function
    loss_function = Loss_CCE()

    # Helper variables
    lowest_loss = 999_999
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    '''
    Training
    '''

    for iteration in range(100_000):
        '''
        Updating weights
        '''

        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        '''
        Forward passing
        '''

        # Pass data through first layer
        dense1.forward(x)
        activation1.forward(dense1.output)

        # Pass data through second layer
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        '''
        Calculating
        '''

        # Calculate loss
        loss = loss_function.calculate(activation2.output, y)

        # Calulate accuracy
        predicitons = np.argmax(activation2.output, axis=1)
        if len(y.shape) == 2:       # Convert from one-hot to categorical labels
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predicitons==y)

        '''
        Outputting
        '''

        if loss < lowest_loss:
            print(f"New set of weights found! Iteration: {iteration}, Loss: {loss}, Accuracy: {accuracy}")
            lowest_loss = loss
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()