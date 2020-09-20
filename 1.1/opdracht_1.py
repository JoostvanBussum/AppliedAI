import numpy as np

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

# This function calculates the distance between two plotted feature vector according to pythagoras' theorem
def calculatePlotDistance(vector1, vector2):
    distance = 0

    # Apply pythagoras to 7 dimensional vector. Add total distance together
    for i in range(len(vector1)):
      distance += ((vector1[i] - vector2[i]) ** 2)
    
    # Return distance squared
    return np.sqrt(distance)

# Decide which label is most common in the list and returns it
def decideLabel(labelList):
  labelCountList = [[0, "winter"], [0, "zomer"], [0, "herfst"], [0, "lente"]]

  # The loop that counts what label occurs what amount of times
  for label in labelList:
    for counter in labelCountList:
      if label[1] == counter[1]:
        counter[0] += 1
        break
  
  # Sort the label count list from highest to lowest
  # in case of a tie the sorting algorith has the last say in which it puts on index 0 
  labelCountList.sort(reverse=True, key=lambda tup: tup[0])
  return labelCountList[0][1]

# Finds k nearest neighbours for value K in given parameters. Returns an array filled with tuples with coupled predicted label and the given correct label
def kNN(classifierSet, classifierSetLabels, validationData, validationDataLabels, k):

  # Protection against invalid values of K
  if k > 0:
    if k < len(classifierSet):

      # Storage for label tuples
      validatedLabels = []

      # Create plot for every day/entry in validationData
      for i in range(len(validationData)):

        # Distance 2 dimensional array contains 2 elements: [[plotDistance, predictedLabel]]
        distance = []
        for j in range(len(classifierSet)):
          # Calculate the distance between the entry plot vector and the classifier plot vector
          tempDistance = calculatePlotDistance(validationData[i], classifierSet[j])
          distance.append([tempDistance, classifierSetLabels[j]])
                
        # Sort and slice the list
        distance.sort(key= lambda tup: tup[0])
        distance = distance[:k]

        # Append tuple to validated label array: [predicted label, true label]
        validatedLabels.append([decideLabel(distance), validationDataLabels[i]])
      
      return validatedLabels    

    raise Exception("K cannot be bigger than classifierSet")
  raise Exception("K cannot be zero or negative")

# Function which finds the optimal value of K based on the algorithm function kNN
def findOptimalK(classifierSet, classifierSetLabels, validationData, validationDataLabels):

  # Array of tuples containing value of [k, percentageCorrect]
  outcome = []
  correctCounter = 0
  
  # For a max k of 100, calculate which k has the highest percentage correct predictions
  for i in range(1, 100):
    # Correctly predicted outcomes
    correctCounter = 0

    # List of labels containing predicted labels and true labels
    labelList = kNN(classifierSet, classifierSetLabels, validationData, validationDataLabels, i)

    # Loop that checks whether the prediction is correct and adds to correctCounter if true
    for j, k in labelList:
      if j == k:
        correctCounter += 1
      
    # Calculate the percentage of correct predictions
    kPercentageCorrect = (correctCounter/100) * 100
    outcome.append([i, kPercentageCorrect])
  
  # Sort the outcome list, splice and return 3 best values of K
  outcome.sort(reverse=True, key= lambda tup: tup[1])
  outcome = outcome[:3]
  return outcome

# void main() {
topK3 = findOptimalK(classifierData, classifierLabels, validationSet, validationLabels)
print(topK3)

# return 0;
# }
