#######################################
#Neural Network Main Module
#NeuralNetwork.py
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

#Import necessary modules
import random

#Constant Variables
const_e = 2.7182818284590452353602874713527

#Activation Function (defaults to the sigmoid function)
def act(x):
    if (abs(x) < 32):
        return 1/(1 +  const_e**-x)
    elif (x > 0):
        return 1
    else:
        return 0

#The derivitive of the activation function
def dirAct(x):
    if (abs(x) < 32):
        return const_e**x/((1+const_e**x)**2)
    else:
        return 0

#Main Neural Network class
class NeuralNetwork:
	
	#Initilizer 
	#Takes ( Array of the number of neurons in each layer , the activation function, the derivitive of the activation function)
    def __init__(self,numLayers,actFunction = act, actDir = dirAct):
        self.layers = []
        self.weights = {}
        self.neurons = []
        self.cach = {}
        self.canCach = False
        self.act = actFunction
        self.dirAct = actDir
       	
        global act 
        act = self.act
        
        global dirAct
        dirAct = self.dirAct
        
	#Initilizes the neurons
        layerOn = 0
        for numInCurrentLayer in numLayers:
            self.layers.append([])
            
            for i in range(numInCurrentLayer):
                neuronOn = len(self.neurons)
                
                self.neurons.append(Neuron(layerOn, neuronOn))
                self.layers[layerOn].append(neuronOn)
                
            layerOn += 1
                
	#Initilizes Weights
        layerOn = 0
        for layer in self.layers:
            if (layerOn != len(self.layers) - 1):
                for n1 in layer:
                    for n2 in self.layers[layerOn + 1]:
                        self.weights[(n1,n2)] = random.randint(-100,100)/100.0
            
            layerOn += 1
    
	#Returns the values of the neuron calculations
    def calcNeuron(self,neuronIndex,inputs):
        return self.neurons[neuronIndex].calc(inputs,self.weights,self.layers,self.neurons)
    
	#Trains the network based upon an expected output
    def train(self,inputs,expectedOut):
        error = []
		
		#Resets the cach of neuron values
        for n in self.neurons:
            n.canCach = False
		
		#Calculates the error values
        outLayerIndex = 0
        for expect in expectedOut:
			error.append(expect - self.calcNeuron(self.layers[len(self.layers) - 1][outLayerIndex],inputs))
			outLayerIndex += 1
        
		#Actualy manipulates the weights
        numDone = 0
        for key in self.weights.keys():
            totalChange = 0
            i = 0
			
			#Averages the changes required
            for outIndex in self.layers[len(self.layers) - 1]:
                d = self.calcDir(inputs,key,outIndex,error[i])
                if (d != 0):
                    totalChange += error[i]/d
                i += 1
                
            totalChange /= len(self.layers[len(self.layers) - 1])
            
			#Applies the changes
            self.weights[key] += totalChange/2        #LEARNING CONST
            
            error = []
    
            outLayerIndex = 0
            for expect in expectedOut:
                error.append(expect - self.calcNeuron(self.layers[len(self.layers) - 1][outLayerIndex],inputs))
                outLayerIndex += 1
    
	#Calculates the result of the neural network's calculations given weights
    def calc(self,inputs):
        for n in self.neurons:
            n.canCach = False
        retArray = []
        for outNeuronIndex in self.layers[len(self.layers) - 1]:
            retArray.append(self.neurons[outNeuronIndex].calc(inputs,self.weights,self.layers,self.neurons))
        
        return retArray
    
	#Calculates the second part of the derivative of the weights with respect to the error
    def calcDown(self, neuronIndex,outIndex,error):
        if self.canCach:
            try:
                return self.cach[(neuronIndex,outIndex,error)]
            finally:
                pass
        for n in self.neurons:
			n.canCach = False
        if neuronIndex in self.layers[len(self.layers) - 1]:
            if neuronIndex == outIndex:
                return dirAct(error)
            else:
                return 0
        else:
            retVal = 0
            
            for indexNextLayer in self.layers[self.neurons[neuronIndex].layerNumber + 1]:
                retVal += self.weights[(neuronIndex,indexNextLayer)] * self.calcDown(indexNextLayer,outIndex,error)
                
            self.cach[(neuronIndex,outIndex,error)] = dirAct(retVal)
            
            return dirAct(retVal)
    
	#Calculates the derivative of a given weight with respect to the error
    def calcDir(self,inputs,(weightIndex1,weightIndex2),outIndex,error):
        self.canCach = False
        self.cach = {}
        return self.calcNeuron(weightIndex1,inputs) * self.calcDown(weightIndex2,outIndex,error)

#Individual Neuron class
class Neuron:
    def __init__(self,layerNumber, index):
        self.layerNumber = layerNumber
        self.index = index
        self.canCach = False
        self.prevValue = 0
    
	#Calculates the value of the neuron
    def calc(self,inputs,weights,layers,neurons):
		#Checks if it can cach the result
        if self.canCach:
			return self.prevValue     
		#Checks if the neuron is an input
        elif self.layerNumber == 0:
            return inputs[self.index]
        else:
			#Sums over all of the inputs to the neuron
            total = 0
            for neuronIndex in layers[self.layerNumber - 1]:
                total += weights[(neuronIndex,self.index)] * neurons[neuronIndex].calc(inputs,weights,layers,neurons)
            
			#Caches the value
            self.prevValue = act(total)
            self.canCach = True
            return act(total)
