# our genotype will be a list that looks as follows(This is an example): [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 
# Every position in the list represents one of the numbers(it being equal to index+1) and the 0 or 1 represents in what pile the number is allocated to
# This way we can easily perform crossovers and mutations on the genotypes

import numpy
import copy
import math
import random

class geneticIndividual:
    def __init__(self):
        self.fitness = 0
        self.genotype = []



class evolutionaryAlgorithm:
    self.population = []

    def __init__(self, populationSize =100, numberOfGenerations =100, genotypeLength =10):
        self.populationSize = populationSize
        self.genotypeLength = genotypeLength



    def generatePopulation(self):
        self.population.append(geneticIndividual)
    
    def crossover(self):
        continue

    def mutate(self):
        continue
        
