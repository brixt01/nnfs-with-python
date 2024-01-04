import numpy as np
import nnfs

from layers import *
from activations import *
from losses import *
from combinations import *
from optimisers import *

from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    '''
    Initlialise data
    '''

    # Create dataset
    X, y = nnfs.spiral_data(samples=100, classes=3)

    '''
    Initialise neural network
    '''

    dense1 = Layer_Dense(2, 64)     # Dense layer: 2 inputs, 64 neurons
    activation1 = Activation_ReLU()     # ReLU activation
    dense2 = Layer_Dense(64, 3)     # Dense layer: 64 inputs, 3 neurons
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()     # Softmax classifier, with loss function
    optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)      # Adaptive momentum optimiser
    
    '''
    Training loop
    '''
    
    for epoch in range(10001):

        '''
        Forward pass
        '''

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        '''
        Stats calculations
        '''

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
            print(f' \
                  epoch: {epoch} \
                  acc: {accuracy:.3f} \
                  loss: {loss:.3f} \
                  lr: {optimizer.current_learning_rate} \
            ')
        
        '''
        Backward pass
        '''

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        '''
        Update weights/biases
        '''

        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()