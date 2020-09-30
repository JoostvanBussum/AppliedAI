import random
import math
import numpy as np

random.seed(0)

def calculateSigmoid(output):

def calculateSigmoidDerivative(output):

# Very very customizable neural network :)
class NeuralNetwork:

    def __init__(self, inputLayerSize, outputLayerSize, isSigmoidNetwork = True, learningRate = 0.1):

    # Runs the network once with the networkInput as input value for the inputneurons. Expects a list the size of the inputvalues. 
    # Will generate an error if the networkInput is smaller or larger than the inputLayer
    def generateNetworkOutputOnce(self, networkInput):

    # Updates all the errors in the entire network based on the desiredOutput. Network needs to be ran at least once before calling this function. 
    def __backpropagation(self, desiredOutput):

    # Updates the entire network. Should only be ran after calling backpropagation
    def __updateNetwork(self):  
    
    # gelijk aan uitvoerlayer size - invoerData: [[1, 2, 3, 4], [1, 3, 2, 4], []]
    # gelijk aan uitvoerlayer size - trainingDataDesiredOutput: [[0.567, 0.023, 0.490], []]
    # Trainingdata: [[input values],[truth values]] ex: [ [ [1, 0], [0, 1] ], [ [1], [1] ] ]
    def trainNetwork(self, trainingDataInput, trainingDataDesiredOutput, trainingIterations = 10000):
          
    # Sets weight and threshold of targeted neuron. 
    # Layer is a boolean: 1 is the hiddenLayer and 0 the outputLayer
    # The function parameter: 'neuron' contains the index of the neuron of the chosen layer. Neuron layer array indexing starts at 0 (Because arrays always start at 0)
    def calibrateNeuron(self, neuron, layer, newWeights, newThreshold = None):

    # Print the network for debug purposes
    def printNetwork(self):

class Neuron:
    # Neuron initialiser with inputs(which is its input neurons) and its threshold
    def __init__(self, isSigmoid = True, weights = None, threshold = None, bias = None, previousLayer = None):

    # Prints all values of the neuron
    def printNeuron(self):

    # Used to generate an output using a single neuron instead of a network
    # Inputs should be remembered outside of neuron scope.
    # Neuron stores the output it generates from the given input based on if it is a perceptron or a sigmoid neuron
    # inputs is an array with inputs
    # Only used for excercise 4.3 A and 4.3 B
    def generateSingleNeuronOutput(self, inputs):

    # Calculates new output based on outputs of all neurons in the previous layer based on their weight
    def feedForward(self):

    # Updates the neuron's weights and bias. Error and output should be updated before this function is called.
    def update(self, learnRate):

    # Getter function for the __neuronInput variable
    def getNeuronInput(self):

    # Sets the error of the neuron. Error is calculated in the network based on whether neuron is placed in hidden or output layer
    def setError(self, value):

    # Sets the weights of the neuron. Parameter expects a list
    def setWeight(self, newWeights):

    # Sets the threshold of the neuron
    def setThreshold(self, newThreshold):

 
    # Generates the new output, modifies variable to equate to new output and returns the newly generated value
    def getOutput(self):

    # Getter function for the __error variable
    def getError(self):

    # Getter function for the __weights variable, if no index is given the entire list is returned
    def getWeights(self, index):

    # Used to set up input layer in neural network by injecting dataset values into neurons
    # to allow undefined network sizes to still generically retrieve outputs with feedforward looping
    def setOutput(self, value):


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
