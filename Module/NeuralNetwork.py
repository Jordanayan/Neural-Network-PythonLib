#######################################
#Neural Network Main Module
#NeuralNetwork.py
#
#(c) 2017 Carter N. Plasek
#######################################

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
                
        print self.layers
                
        layerOn = 0
        for layer in self.layers:
            if (layerOn != len(self.layers) - 1):
                for n1 in layer:
                    for n2 in self.layers[layerOn + 1]:
                        self.weights[(n1,n2)] = random.randint(-100,100)/100.0
            
            layerOn += 1
            
        print self.weights
        
    def calcNeuron(self,neuronIndex,inputs):
        return self.neurons[neuronIndex].calc(inputs,self.weights,self.layers,self.neurons)
    
    def train(self,inputs,expectedOut):
        error = []
		
        for n in self.neurons:
            n.canCash = False

        outLayerIndex = 0
        for expect in expectedOut:
			error.append(expect - self.calcNeuron(self.layers[len(self.layers) - 1][outLayerIndex],inputs))
			outLayerIndex += 1
            
        numDone = 0
        for key in self.weights.keys():
            totalChange = 0
            i = 0
            for outIndex in self.layers[len(self.layers) - 1]:
                d = self.calcDir(inputs,key,outIndex,error[i])
                if (d != 0):
                    totalChange += error[i]/d
                i += 1
                
            totalChange /= len(self.layers[len(self.layers) - 1])
            
            self.weights[key] += totalChange/2        #LEARNING CONST
            
            error = []
    
            outLayerIndex = 0
            for expect in expectedOut:
                error.append(expect - self.calcNeuron(self.layers[len(self.layers) - 1][outLayerIndex],inputs))
                outLayerIndex += 1
            
    def calc(self,inputs):
        for n in self.neurons:
            n.canCash = False
        retArray = []
        for outNeuronIndex in self.layers[len(self.layers) - 1]:
            retArray.append(self.neurons[outNeuronIndex].calc(inputs,self.weights,self.layers,self.neurons))
        
        return retArray
    
    def calcDown(self, neuronIndex,outIndex,error):
        if self.canCach:
            try:
                return self.cach[(neuronIndex,outIndex,error)]
            finally:
                pass
        for n in self.neurons:
			n.canCash = False
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
    
    def calcDir(self,inputs,(weightIndex1,weightIndex2),outIndex,error):
        self.canCach = False
        self.cach = {}
        return self.calcNeuron(weightIndex1,inputs) * self.calcDown(weightIndex2,outIndex,error)
    
class Neuron:
    def __init__(self,layerNumber, index):
        self.layerNumber = layerNumber
        self.index = index
        self.canCash = False
        self.prevValue = 0
    
    def calc(self,inputs,weights,layers,neurons):
        if self.canCash:
			return self.prevValue        
        elif self.layerNumber == 0:
            return inputs[self.index]
        else:
            total = 0
            for neuronIndex in layers[self.layerNumber - 1]:
                total += weights[(neuronIndex,self.index)] * neurons[neuronIndex].calc(inputs,weights,layers,neurons)
            
            self.prevValue = act(total)
            self.canCash = True
            return act(total)
