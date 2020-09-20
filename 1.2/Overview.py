import numpy as np
import random
import copy
import time


  
# This function calculates the distance between two plotted feature vector according to pythagoras' theorem
def calculatePlotDistance(vector1, vector2):

# Function that finds the nearest centroid
def findNearestCentroid(plotPointVectorData, centroidList):

# Decide which label is most common in the list and returns it
def decideLabel(cluster):

# The actual kMeans algorithm function
def kMeans(plotPointVector, kCentroids):

# Calculates the intre-cluster distance
def calculateIntraclusterDistance(clusterGraph):

# Function for the screeplot
def screePlot(plotPointVector, kMeansIterationValue):


# Main #
#=============================================================================================================================================================================================================================

random.seed(0)

plotData = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
plotLabels = []
for label in dates:
  if label < 20000301:
    plotLabels.append("winter")
  elif 20000301 <= label < 20000601:
    plotLabels.append("lente")
  elif 20000601 <= label < 20000901:
    plotLabels.append("zomer")
  elif 20000901 <= label < 20000201:
    plotLabels.append("herfst")
  else: # from 01-12 to end of year
    plotLabels.append("winter")

#=============================================================================================================================================================================================================================

# Create array with tuples pairing labels and vectordata: [[Label,[1, 2, 3, 4, 5, 6, 7]]]
plotPointList = []
for i in range(len(plotData)):
  plotPointList.append([plotLabels[i], plotData[i]])

#plotPointList = plotPointList[:100]
screePlotOutcome = screePlot(plotPointList, 5)

print(screePlotOutcome)