# our genotype will be a list that looks as follows(This is an example): [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 
# Every position in the list represents one of the numbers(it being equal to index+1) and the 0 or 1 represents in what pile the number is allocated to(0 being the sum pile and 1 being the multiply pile)
# This way we can easily perform crossovers and mutations on the genotypes

import copy
import math
import random

import numpy

random.seed(0)
class geneticIndividual:

    # Init function that can randomly assign values to the genotype or use a given genotype
    # also calculates the fitness upon being created
    def __init__(self, genotypeLength =10, genotype =None):
    
    # Function to print the genetic individual
    def printGeneticIndividual(self):

    # Function to determine the fitness of an individual
    def evaluateFitness(self):
        
class evolutionaryAlgorithm:

    def __init__(self, populationSize =100, numberOfGenerations =100, genotypeLength =10, mutateChance =50, retainModifier =0.2):

    # Function that randomly generates a starting population
    def generatePopulation(self):
            
    # Function to print the population
    def printPopulation(self):
    
    # Evolve function that first selects parents and then does crossover and mutate
    def evolve(self):
        
    # Single point crossover function with the point being random
    def crossover(self, parents):

    # Function to mutate a population, the mutateChance is in percentage
    # We mutate by swapping one value to the other pile
    def mutate(self, individual):

    # Function that runs the whole algorithm an amount of times equal to generations
    def run(self):

answerCounter = 0
averageFitness = 0
for i in range(100):
    algoritme = evolutionaryAlgorithm()
    algoritme.generatePopulation()
    result = algoritme.run()
    averageFitness += result[0].fitness
    result[0].printGeneticIndividual()
    if result[0].fitness == 0.0:
        answerCounter += 1

averageFitness = averageFitness/100

print(answerCounter, " , ", averageFitness)