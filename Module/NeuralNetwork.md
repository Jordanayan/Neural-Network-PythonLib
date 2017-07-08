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

## License Agreement:

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to:

 The Free Software Foundation, Inc.,
 59 Temple Place - Suite 330,
 Boston, MA  02111-1307, USA.

---

(c) 2017 Carter N. Plasek
