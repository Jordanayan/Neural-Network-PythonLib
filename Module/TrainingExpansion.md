# Neural Network Module

A module to add better training to the neural network module

### Authors:

*Carter N. Plasek*

### License:

GPL v. 3

## TrainingExpansion  
TrainingExpansion.py - allows for more advanced training methods

## Usage

For exmple the following code snippet:

```python
import NeuralNetwork

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

def train():
    for d in data:
        if (len(d[0]) == len(NN.layers[0]) and len(d[1]) == len(NN.layers[len(NN.layers) - 1])):
            NN.train(d[0],d[1])
```

to simply:

```python
import NeuralNetwork
import TrainingExpansion

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

t = TrainingExpansion.Trainer(NN)
t.trainingData = data

t.trainAll()
```

Both of these code examples trains a neural network on the data given (for the curious it trains the network to calculate if the first input value is greater than the second input value). The second example implements the TrainingExpansion module. 

This is not the only way to train the network (on the full data set, which can be rather time intensive if the set or the network is large). The TrainingExpansion module adds the ability to (in a single line of code) train the network on a random subset of the data. 

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

while with the TrainingExpansion module it becomes:

```python
import random
import NeuralNetwork
import TrainingExpansion

NN = NeuralNetwork.NeuralNetwork([2,3,4,1])
data = [[[2,3],[0]],[[5,3],[1]],[[3,3],[0.5]],[[8,9],[0]],[[8,-1],[1]],[[2,-1],[1]],[[-5,-6],[1]]]

t = TrainingExpansion.Trainer(NN)
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
