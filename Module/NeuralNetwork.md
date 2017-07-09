# Neural Network Module

A module to add Neural Network functonality to python code

### Authors:

*Carter N. Plasek*

### License:

GPL v. 3

## NeuralNetwork
NeuralNetwork.py - allows the following example code to be run

## Usage

### Basic Training
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

### Advanced Training

Another feature granted by this module is allowing for the creation of "Trainers" (a wrapper for a Neural Network allowing for more advanced training to take place). For example:

```python
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

def train():
    for d in data:
        if (len(d[0]) == len(NN.layers[0]) and len(d[1]) == len(NN.layers[len(NN.layers) - 1])):
            NN.train(d[0],d[1])
```

Can be rewritten as

```python
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

t = NeuralNetwork.Trainer(NN)
t.trainingData = data

t.trainAll()
```

Both of these code examples trains a neural network on the data given (for the curious it trains the network to calculate if the first input value is greater than the second input value). The second example implements the Advanced Training functionality. 

This is not the only way to train the network (on the full data set, which can be rather time intensive if the set or the network is large). The Trainers also adds the ability to (in a single line of code) train the network on a random subset of the data. 

For Example, the following trains the network on 30% of the total training data.

```python
import random
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

def train():
    for i in range((len(data) - 1) * 0.3):
        d = random.choice(data)
        if (len(d[0]) == len(NN.layers[0]) and len(d[1]) == len(NN.layers[len(NN.layers) - 1])):
            NN.train(d[0],d[1])
```

while with Advanced Training it becomes:

```python
import random
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

t = NeuralNetwork.Trainer(NN)
t.trainingData = data

t.trainPercent(30)
```

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
