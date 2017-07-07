# Neural-Network-PythonLib
A python module to add the ability to create and train neural networks.

# Example Usage #
```python
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,2]) #The number of neurons in each of the layers

def trainNetwork(inputs, expectedOutputs):
        NN.train(inputs,outputs)  #WARNING Spaming this will cause intense lag!

def getOutput(inputs):
        return NN.calc(inputs)
```

This example program shows the base functionality of the Neural Network module. It demonstrates how to create, train, and use the Neural Networks in the simplest method.
