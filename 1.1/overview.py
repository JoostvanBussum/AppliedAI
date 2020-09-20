import numpy as np

# This function calculates the distance between two plotted feature vector according to pythagoras' theorem
def calculatePlotDistance(vector1, vector2):

# Decide which label is most common in the list and returns it
def decideLabel(labelList):

# Finds k nearest neighbours for value K in given parameters. Returns an array filled with tuples with coupled predicted label and the given correct label
def kNN(classifierSet, classifierSetLabels, validationData, validationDataLabels, k):

# Function which finds the optimal value of K based on the algorithm function kNN
def findOptimalK(classifierSet, classifierSetLabels, validationData, validationDataLabels):

classifierData = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
classifierLabels = []
for label in dates:
  if label < 20000301:
    classifierLabels.append("winter")
  elif 20000301 <= label < 20000601:
    classifierLabels.append("lente")
  elif 20000601 <= label < 20000901:
    classifierLabels.append("zomer")
  elif 20000901 <= label < 20000201:
    classifierLabels.append("herfst")
  else: # from 01-12 to end of year
    classifierLabels.append("winter")

validationSet = np.genfromtxt("validation1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

validationDates = np.genfromtxt("validation1.csv", delimiter=";", usecols=[0])
validationLabels = []
for label in validationDates:
  if label < 20010301:
    validationLabels.append("winter")
  elif 20010301 <= label < 20010601:
    validationLabels.append("lente")
  elif 20010601 <= label < 20010901:
    validationLabels.append("zomer")
  elif 20010901 <= label < 20010201:
    validationLabels.append("herfst")
  else: # from 01-12 to end of year
    validationLabels.append("winter")

# Data used to train find K
days = np.genfromtxt("days.csv", delimiter=";", usecols=[1,2,3,4,5,6,7])

topK3 = findOptimalK(classifierData, classifierLabels, validationSet, validationLabels)
print(topK3)
