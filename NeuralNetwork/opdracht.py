import random

random.seed(0)

def class NeuralNetwork(self):

    def __init__(self, inputLayerSize, outputLayerSize):
        
        #self.hiddenLayer = []
        #self.outputLayer = [append()for index in range(len(outputLayerSize))]

    def addNeuron(self, neuron, layer):
        self.neurons[][layer-1].append(neuron)
        

def class Neuron(self):
    # Neuron initialiser with inputs(which is its input neurons) and its threshold
    def __init__(self, previousLayer, weights, threshold = None):

        # List with connected neurons one layer before this neuron
        self.previousLayer = previousLayer
        self.weights = weights

        self.threshold = treshold
        self.output = 0
    
    # Calculates new output based on outputs of all neurons in the previous layer based on their weight
    def feedForward(self):
        # Temp variable for the output
        tempOutput = 0
        
        for i in range(len(self.previousLayer)):
            tempOutput += previousLayer[i].getOutput() * weights[i]

        # Perceptron threshold checking and setting output
        if tempOutput >= self.threshold:
            self.output = 1
        else:
            self.output = 0

        return self.output

    # Generates the new output, modifies variable to equate to new output and returns the newly generated value
    def getOutput(self):

        return self.output

    # Used to set up input layer in neural network by injecting dataset values into neurons
    # to allow undefined network sizes to still generically retrieve outputs with feedforward looping
    def setOutput(self, value):
        self.output = value

neuron1 = neuron()
neuron2 = [0]
neuron3 = neuron(list[[neuron1, neuron1.getOwnWeight], [neuron2, neuron2.getOwnWeight]])