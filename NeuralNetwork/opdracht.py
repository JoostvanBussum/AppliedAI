import random
import math
import numpy as np

random.seed(0)

def calculateSigmoid(output):
    #print("sigmoid input: ", output)
    #print("sigmoid result: ", (1 / (1 + (math.e ** (-output)))))
    return (1 / (1 + (math.e ** (-output))))

def calculateSigmoidDerivative(output):
    logisticOutput = calculateSigmoid(output)
    return (logisticOutput * (1 - logisticOutput))

# Very very customizable neural network :)
class NeuralNetwork:

    def __init__(self, inputLayerSize, outputLayerSize, isSigmoidNetwork = True, learningRate = 0.1):

        self.learnRate = learningRate
        self.isSigmoidNetwork = isSigmoidNetwork

        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize

        self.inputLayer  = []
        self.hiddenLayer = []
        self.outputLayer = []

        # Initialize inputlayer with "fake" neurons so we can input data into the network
        for i in range(self.inputLayerSize):
            self.inputLayer.append(Neuron())
        
        # Initialize hidden layer
        for i in range(self.inputLayerSize):
            
            # Create list with random weights equal to the size of the inputlayerSize
            neuronWeights = []
            for j in range(self.inputLayerSize):
                neuronWeights.append(random.uniform(-1,1))

            self.hiddenLayer.append(Neuron(isSigmoidNetwork, neuronWeights, random.uniform(-1,1), random.uniform(-1,1), self.inputLayer))
        
        # Initialize output layer
        for i in range(self.outputLayerSize):   
            
            # Create list with random weights equal to the size of the hiddenLayerSize      
            neuronWeights = []
            for j in range(self.inputLayerSize):
                neuronWeights.append(random.uniform(-1,1))
                
            self.outputLayer.append(Neuron(isSigmoidNetwork, neuronWeights, random.uniform(-1,1), random.uniform(-1,1), self.hiddenLayer))


    # Runs the network once with the networkInput as input value for the inputneurons. Expects a list the size of the inputvalues. 
    # Will generate an error if the networkInput is smaller or larger than the inputLayer
    def generateNetworkOutputOnce(self, networkInput):
        networkOutput = []

        # Inputs the given parameters 
        if len(networkInput) == len(self.inputLayer):
            for i in range(len(self.inputLayer)):
                self.inputLayer[i].setOutput(networkInput[i])

            for i in range(len(self.hiddenLayer)):
                self.hiddenLayer[i].feedForward()

            for i in range(len(self.outputLayer)):
                networkOutput.append(self.outputLayer[i].feedForward())

            return networkOutput

        else:
            raise Exception("networkInput cannot be of a different size than the inputLayer of the neural network")


    # Updates all the errors in the entire network based on the desiredOutput. Network needs to be ran at least once before calling this function. 
    def __backpropagation(self, desiredOutput):

        # Updates all the error values for the outputLayer
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].setError((desiredOutput[i] - self.outputLayer[i].getOutput()) * calculateSigmoidDerivative(self.outputLayer[i].getNeuronInput()))
        
        for i in range(len(self.hiddenLayer)):
            sumError = 0.0
            
            for j in range(len(self.outputLayer)):
                sumError += float(self.outputLayer[j].getError()) * self.outputLayer[j].getWeights(i)
            
            self.hiddenLayer[i].setError(calculateSigmoidDerivative(self.hiddenLayer[i].getNeuronInput()) * sumError)    


    # Updates the entire network. Should only be ran after calling backpropagation
    def __updateNetwork(self):
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].update(self.learnRate)
        
        for i in range(len(self.hiddenLayer)):
            self.hiddenLayer[i].update(self.learnRate)
    
    
    # gelijk aan uitvoerlayer size - invoerData: [[1, 2, 3, 4], [1, 3, 2, 4], []]
    # gelijk aan uitvoerlayer size - trainingDataDesiredOutput: [[0.567, 0.023, 0.490], []]
    # Trainingdata: [[input values],[truth values]] ex: [ [ [1, 0], [0, 1] ], [ [1], [1] ] ]
    def trainNetwork(self, trainingDataInput, trainingDataDesiredOutput, trainingIterations = 10000):
        
        for i in range(trainingIterations):
            
            for trainingExampleIndex in range(len(trainingDataInput)):
                self.generateNetworkOutputOnce(trainingDataInput[trainingExampleIndex])
                self.__backpropagation(trainingDataDesiredOutput[trainingExampleIndex])
                self.__updateNetwork()
            

    # Sets weight and threshold of targeted neuron. 
    # Layer is a boolean: 1 is the hiddenLayer and 0 the outputLayer
    # The function parameter: 'neuron' contains the index of the neuron of the chosen layer. Neuron layer array indexing starts at 0 (Because arrays always start at 0)
    def calibrateNeuron(self, neuron, layer, newWeights, newThreshold = None):
        
        if layer and neuron < len(hiddenLayer):
            self.hiddenLayer[neuron].setWeight(newWeights)
            if newThreshold:
                self.hiddenLayer[neuron].setThreshold(newThreshold)

        elif not layer and neuron < len(outputLayer):
            self.outputLayer[neuron].setWeight(newWeights)
            self.outputLayer[neuron].setThreshold(newThreshold)
            
            if newThreshold:
                self.outputLayer[neuron].setThreshold(newThreshold)


    # Print the network for debug purposes
    def printNetwork(self):
        print("\n Inputlayer: \n")
        for i in range(len(self.inputLayer)):
            self.inputLayer[i].printNeuron()

        print("\n Hiddenlayer: \n")
        for i in range(len(self.hiddenLayer)):
            self.hiddenLayer[i].printNeuron()

        print("\n Outputlayer: \n")
        for i in range(len(self.outputLayer)):
            self.outputLayer[i].printNeuron()


class Neuron:
    # Neuron initialiser with inputs(which is its input neurons) and its threshold
    def __init__(self, isSigmoid = True, weights = None, threshold = None, bias = None, previousLayer = None):

        # List with connected neurons one layer before this neuron
        self.previousLayer = previousLayer
        self.__weights = weights

        self.__error = 0
        self.__bias = bias
        self.__threshold = threshold
        self.__output = 0
        self.__neuronInput = 0
        self.__isSigmoid = isSigmoid


    # Prints all values of the neuron
    def printNeuron(self):
        print("\n Neuron: ")
        print("previousLayer: ", self.previousLayer)
        print("Weights: ", self.__weights)
        print("Threshold: ", self.__threshold)
        print("Bias: ", self.__bias)
        print("Output: ", self.__output)


    # Used to generate an output using a single neuron instead of a network
    # Inputs should be remembered outside of neuron scope.
    # Neuron stores the output it generates from the given input based on if it is a perceptron or a sigmoid neuron
    # inputs is an array with inputs
    # Only used for excercise 4.3 A and 4.3 B
    def generateSingleNeuronOutput(self, inputs):

        if len(inputs) == len(self.__weights):
            self.__neuronInput = 0.0

            for i in range(len(inputs)):
                self.__neuronInput += inputs[i] * self.__weights[i]
                
            if self.__isSigmoid:
                self.__neuronInput += self.__bias
                self.__output = calculateSigmoid(self.__neuronInput)
                return self.__output
             
            else:
                if self.__neuronInput >= self.__threshold:
                    self.__output = 1
                else:
                    self.__output = 0
                    
            return self.__output
        
        raise Exception("Number of inputs does not match the number of weights given")
        

    # Calculates new output based on outputs of all neurons in the previous layer based on their weight
    def feedForward(self):
        self.__neuronInput = 0.0

        for i in range(len(self.previousLayer)):
            self.__neuronInput += self.previousLayer[i].getOutput() * self.__weights[i]

        # Perceptron threshold checking and setting output
        if self.__isSigmoid:
            self.__neuronInput += self.__bias
            self.__output = calculateSigmoid(self.__neuronInput)

        else:
            if self.__neuronInput >= self.__threshold:
                self.__output = 1
            else:
                self.__output = 0

        return self.__output


    # Updates the neuron's weights and bias. Error and output should be updated before this function is called.
    def update(self, learnRate):
        
        for i in range(len(self.__weights)):
            self.__weights[i] += learnRate *  self.__error * self.previousLayer[i].getOutput()

        self.__bias += learnRate * self.__error


    # Getter function for the __neuronInput variable
    def getNeuronInput(self):
        return self.__neuronInput


    # Sets the error of the neuron. Error is calculated in the network based on whether neuron is placed in hidden or output layer
    def setError(self, value):
        self.__error = value


    # Sets the weights of the neuron. Parameter expects a list
    def setWeight(self, newWeights):
        self.__weights = newWeights


    # Sets the threshold of the neuron
    def setThreshold(self, newThreshold):
        self.__threshold = newThreshold

 
    # Generates the new output, modifies variable to equate to new output and returns the newly generated value
    def getOutput(self):
        return self.__output

    # Getter function for the __error variable
    def getError(self):
        return self.__error


    # Getter function for the __weights variable, if no index is given the entire list is returned
    def getWeights(self, index):
        return self.__weights[index]

    # Used to set up input layer in neural network by injecting dataset values into neurons
    # to allow undefined network sizes to still generically retrieve outputs with feedforward looping
    def setOutput(self, value):
        self.__output = value

#================================================================== Main ==============================================================================

# ================================================================== NOR Gate ==================================================================
#norWeights = [-1, -1, -1]
#norNeuron = Neuron(norWeights, 0)
#print("Output NOR Neuron [1, 1, 0]: ", norNeuron.generateSingleNeuronOutput([1, 1, 0]))
#print("Output NOR Neuron [0, 1, 0]: ", norNeuron.generateSingleNeuronOutput([0, 1, 0]))
#print("Output NOR Neuron [0, 1, 1]: ", norNeuron.generateSingleNeuronOutput([0, 1, 1]))
#print("Output NOR Neuron [0, 0, 0]: ", norNeuron.generateSingleNeuronOutput([0, 0, 0]))

#================================================================== Neural Adder ==================================================================
#network = NeuralNetwork(2, 2)

# Delta Rule
#trainingNeuron = Neuron()

#================================================================== XOR ==================================================================
"""
xorData = [ [0,0], [0,1], [1,0], [1,1] ]
xorDataOutput = [ [0], [1], [1], [0] ]

network = NeuralNetwork(2, 1, False, 0.1)
network.printNetwork()

network.trainNetwork(xorData, xorDataOutput, 100000)
network.printNetwork()

tempNetworkOutput1 = network.generateNetworkOutputOnce([0,0])
tempNetworkOutput2 = network.generateNetworkOutputOnce([0,1])
tempNetworkOutput3 = network.generateNetworkOutputOnce([1,0])
tempNetworkOutput4 = network.generateNetworkOutputOnce([1,1])

print("Output 1: ", tempNetworkOutput1)
print("Output 2: ", tempNetworkOutput2)
print("Output 3: ", tempNetworkOutput3)
print("Output 4: ", tempNetworkOutput4)
"""
#================================================================== Bloemen ==================================================================

trainingData = np.genfromtxt("iris.data", delimiter=",", usecols=[0,1,2,3])
trainingDataOutput = np.genfromtxt("iris.data", delimiter=",", usecols=[4], dtype=str)
convertedOutput = []

for i in range(len(trainingDataOutput)):
    if trainingDataOutput[i] == "Iris-setosa":
        convertedOutput.append(list([1,0,0]))
    if trainingDataOutput[i] == "Iris-versicolor":
        convertedOutput.append(list([0,1,0]))
    if trainingDataOutput[i] == "Iris-virginica":
        convertedOutput.append(list([0,0,1]))

irisNetwork = NeuralNetwork(4, 3, True, 0.1)
irisNetwork.trainNetwork(trainingData, convertedOutput, 10000)


while True:
    print("Geef een index van de iris.data set")
    dataSetIndex = input("Vul hier uw waarde in: ")
    dataSetIndex = int(dataSetIndex)
    
    if dataSetIndex >= 0 and dataSetIndex <= len(trainingData-1):
        print("Netwerk output: ", irisNetwork.generateNetworkOutputOnce(trainingData[dataSetIndex]))

    else:
        print("De ingevoerde index lag buiten de grootte van de trainingdataset!")
