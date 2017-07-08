# Neural Network Module

A module to add Neural Network functonality to python code

### Authors:

*Carter N. Plasek*

### License:

GPL v. 3

## NeuralNetwork
NeuralNetwork.py - allows the following example code to be run

## Usage

```python
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,2]) #The number of neurons in each of the layers

def trainNetwork(inputs, expectedOutputs):
        NN.train(inputs,outputs)  #WARNING Spaming this will cause intense lag!

def getOutput(inputs):
        return NN.calc(inputs)
```

The code snippet preceding this is designed to be an example of the usage of the Neural Network Module.
The trainNetwork function trains the network (over time, it is not instantenious) to respond to that input 
with the given output. The getOutput function returns the network's responce to the given input.

===

(c) 2017 Carter N. Plasek
