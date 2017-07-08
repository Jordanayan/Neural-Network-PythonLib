#######################################
#Training Expansion Module
#TrainingExpansion.py
#
#(c) 2017 Carter N. Plasek
#######################################

#######################################
#This program is free software; you can redistribute it and/or modify it under
#the terms of the GNU General Public License as published by the Free Software
#Foundation; either version 3 of the License, or (at your option) any later
#version.

#This program is distributed in the hope that it will be useful, but WITHOUT
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with
#this program; if not, write to:

 #The Free Software Foundation, Inc.,
 #59 Temple Place - Suite 330,
 #Boston, MA  02111-1307, USA.
#######################################

#Imports the Neural Network Module if neccessary
import warnings
if (!_NeuralNetwork_Defined_):
    warnings.warn("Neural Network Module not defined... importing now")
    import NeuralNetwork
  
class Trainer:
    def __init__(self,neuralNetworkLink, hasBias = False):
        self.neuralNetwork = neuralNetworkLink
        self.numInputs = len(neuralNetworkLink.layers[0])
        self.numOutputs = len(neuralNetworkLink.layers[len(neuralNetworkLink.layers) - 1])
        self.hasBias = hasBias
        self.trainingData = []
        
    def trainAll(self):
        for d in self.trainingData:
            if (len(d[1]) == self.numOutputs):
                if (len(d[0]) == self.numInputs):
                    self.neuralNetwork.train(d[0],d[1])
                else:
                    d.append(1)
                    self.neuralNetwork.train(d[0],d[1])
                    
    def trainPercent(self,percent):
        for i in range((len(self.trainingData) - 1)*percent/100):
            d = random.choice(self.trainingData)
            if (len(d[1]) == self.numOutputs):
                if (len(d[0]) == self.numInputs):
                    self.neuralNetwork.train(d[0],d[1])
                else:
                    d.append(1)
                    self.neuralNetwork.train(d[0],d[1])
