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
        self.inputs = inputs
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

    # Backward pass
    def backward(self, dvalues):
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

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

# Softmax activation and CCE loss combination
#
# Combines the functionality of softmax and CCE into one function, which is able
# to perform faster (due to parts of backpropogation cancelling out)
class Activation_Softmax_Loss_CCE():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CCE()
    
    # Forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        num_samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(num_samples), y_true] -= 1
        self.dinputs = self.dinputs / num_samples


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
    loss_activation = Activation_Softmax_Loss_CCE()      # Softmax activation function


    '''
    Forward passing
    '''

    # Pass data through first layer
    dense1.forward(x)
    activation1.forward(dense1.output)

    # Pass data through second layer
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    '''
    Outputting
    '''

    print(loss_activation.output[:5])
    print(f"Loss: {loss}")

    # Calulate accuracy
    predicitons = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:       # Convert from one-hot to categorical labels
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predicitons==y)

    print(f"Acc: {accuracy}")

    '''
    Backward passing
    '''

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    '''
    Outputting
    '''

    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)