import numpy as np
import random
import copy
import time

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
  

def calculatePlotDistance(vector1, vector2):
    distance = 0

    # Apply pythagoras to 7 dimensional vector. Add total distance together
    for i in range(len(vector1)):
      distance += ((vector1[i] - vector2[i]) ** 2)

    # Return distance not squared to make outliers more apparent
    return distance


def findNearestCentroid(plotPointVectorData, centroidList):
  closestCentroid = [centroidList[0][0], calculatePlotDistance(plotPointVectorData, centroidList[0][0][1])]

  for i in range(1, len(centroidList)):
    tempDistance = calculatePlotDistance(plotPointVectorData, centroidList[i][0][1])
    if tempDistance < closestCentroid[1]:
      closestCentroid = [centroidList[i][0], tempDistance]

  return closestCentroid[0]

def decideLabel(cluster):
  labelCountList = [[0, "winter"], [0, "zomer"], [0, "herfst"], [0, "lente"]]

  # The loop that counts what label occurs what amount of times
  for label in cluster:
    for counter in labelCountList:
      if cluster[1][i][0] == counter[1]:
        counter[0] += 1
        break
  
  # Sort the label count list from highest to lowest
  # in case of a tie the sorting algorith has the last say in which it puts on index 0 
  labelCountList.sort(reverse=True, key=lambda tup: tup[0])
  return labelCountList[0][1]

def kMeans(plotPointVector, kCentroids):

  # Bool if centroids changed 
  centroidRepositioned = True

  # Choose k amount of random centroids from plots and initialize as list, first element of cluster is always the centroid!
  centroidList = []
  tempCentroidList = random.sample(copy.deepcopy(plotPointVector), kCentroids)

  # Jank centroid numbering so we can keep array indexing and centroid comparing operations to a minimum
  # Initialize storage list containing clusters with first element of each cluster being the centroid
  # this being indicated by it not having a label but an index instead
  # Length of centroidlist is at this point still equal to kCentroids hence the range length loop
  for i in range(kCentroids):
    tempCentroidList[i][0] = i + 1
    centroidList.append([(tempCentroidList[i])])

  # Container containing newly generated cluster array after each while loop iteration
  newPlotDistributionList = []

  # Loop to find the optimal centroid position for the clustering
  while centroidRepositioned:

    # Opdate break condition of loop which will be flipped if any changes in centroid positions are found
    centroidRepositioned = False

    # Copy the new centroids in a variable to compare later (centroidList may change throughout a while loop iteration)
    tempCentroidList = copy.deepcopy(centroidList)

    # Create a list where all the plots will be appended to the correct centroid for recalculation later
    newPlotDistributionList = copy.deepcopy(centroidList)

    # Iterate over all plot points needing to be assigned to a centroid
    for i in range(len(plotPointVector)):
      nearestCentroid = findNearestCentroid(plotPointVector[i][1], centroidList)

      # Iterate over clusters (centroidlist contains as many centroids as newplotdistributionlist)
      for clusterIndex in range(len(centroidList)):

        # Check if nearestCentroid is equal to the cluster index centroid
        if centroidList[clusterIndex][0][0] == nearestCentroid[0]:

          newPlotDistributionList[clusterIndex].append(plotPointVector[i])
          break

    # Calculate new centroid positions
    for i in range(len(newPlotDistributionList)):

      # Check the size of the cluster of the current centroid. Subtract 1 because centroid is included in the cluster
      clusterSize = len(newPlotDistributionList[i]) - 1
      newCentroidPosition = [[0],[0],[0],[0],[0],[0],[0]]
      
      
      # For each feature vector
      for j in range(1, len(newPlotDistributionList[i])):
        
        # For each feature of the vector calculate the new value
        for k in range(len(newPlotDistributionList[i][j][1])):

          newCentroidPosition[k] += newPlotDistributionList[i][j][1][k]
          
      # Divide all centroid vector values by the cluster size and output into centroidList to finalize the new position of the centroid
      for w in range(len(newCentroidPosition)):
        newVectorValue = np.round((newCentroidPosition[w] / clusterSize), 0)
        centroidList[i][0][1][w] = newVectorValue

      # Check if any changes have ocurred in the position of the current centroid. If any change is found, a centroid has repositioned flipping centroidRepositioned to True
      # if centroidReposition = True : A centroid has repositioned requiring another loop iteration to be done.
      if not centroidRepositioned:
        for currentCentroidVectorIndex in range(len(centroidList[i][0][1])):
          if centroidList[i][0][1][currentCentroidVectorIndex] != tempCentroidList[i][0][1][currentCentroidVectorIndex]:
            centroidRepositioned = True
            break
    
  return newPlotDistributionList


def calculateIntraclusterDistance(clusterGraph):
  totalIntraclusterDistance = 0

  # For every cluster in the clustgraph
  for clusterIndex in range(len(clusterGraph)):

    # Iterate over every plotpoint in the cluster
    for plotPointIndex in range(1, len(clusterGraph[clusterIndex])):
      totalIntraclusterDistance += calculatePlotDistance(clusterGraph[clusterIndex][0][1], clusterGraph[clusterIndex][plotPointIndex][1])
 
  return totalIntraclusterDistance


def screePlot(plotPointVector, kMeansIterationValue):
  intraClusterDistanceCalculationOutcomes = []
  intraClusterDistanceOutcome = []
  kMeansClusterGraph = []

  # Amount of centroids
  for centroidCount in range(2, 15):
      
    intraClusterDistanceCalculationOutcomes = []
    # run kMeans kMeansIterationValue amount of times
    for kMeansIterationIndex in range(kMeansIterationValue):

      # Run kMeans with centroidCount amount of centroids. Output is clusterGraph
      kMeansClusterGraph = kMeans(plotPointVector, centroidCount)

      # Calculate intraClusterDistance from the generated graph
      tempIntraClusterDistance = calculateIntraclusterDistance(kMeansClusterGraph)

      # Append intraClusterDistance to list of outcomes for current centroids
      intraClusterDistanceCalculationOutcomes.append(tempIntraClusterDistance)

    # Pick lowest intraclusterDistance for current amount of centroids and append in list with outcomes
    intraClusterDistanceOutcome.append([centroidCount, min(intraClusterDistanceCalculationOutcomes)])

  return intraClusterDistanceOutcome


# Main #
#=============================================================================================================================================================================================================================

#plotPointList = plotPointList[:100]
screePlotOutcome = screePlot(plotPointList, 5)

print(screePlotOutcome)