'''
Title: Chapter 1
Author: Ben Brixton
'''

if __name__ == "__main__":
    
    inputs = [1.0, 2.0, 3.0, 2.5]

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

    layer_outputs = []      # List of outputs from each neuron

    for neuron_weights, neuron_bias in zip(weights, biases):        # Iterate through each neuron, getting its list of weights and its bias
        neuron_output = 0       # Output of current neuron
        for input, weight in zip(inputs, neuron_weights):       # Iterate through each input/weight pair
            neuron_output += input*weight       # Add input*weight to the neuron's output
        neuron_output += neuron_bias        # Add bias to neuron's output
        layer_outputs.append(neuron_output)     # Add neuron output to list of outputs

    print(layer_outputs)