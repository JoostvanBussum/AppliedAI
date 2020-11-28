# our genotype will be a list that looks as follows(This is an example): [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 
# Every position in the list represents one of the numbers(it being equal to index+1) and the 0 or 1 represents in what pile the number is allocated to(0 being the sum pile and 1 being the multiply pile)
# This way we can easily perform crossovers and mutations on the genotypes

import numpy
import copy
import math
import random

class geneticIndividual:
    def __init__(self):
        self.fitness = 0
        self.genotype = []

    # Function to print genetic individuals
    def printGeneticIndividual(self):
        print("Fitness: ", self.fitness, " genotype: ", self.genotype)


    # Function to determine the fitness of an individual
    def evaluateFitness(self):
        sumPileTotal = 0
        fitnessScore = 0

        # Count how many items are in pile 1, because if it is 1 or more the value needs to start at 1 else we multiply by 0
        if genotype.count(1) >= 1:
            MultiplyPileTotal = 1
        else: 
            MultiplyPileTotal = 0

        # Calculate the total value of the piles
        # Because i starts at 0 and every items value is equal to it's position we do i+1
        # so that item 1 (at index 0) still has a value of 1 and so on for all the items
        for i in range(len(self.genotype)):
            if self.genotype[i] == 0:
                sumPileTotal += i+1
            else:
                MultiplyPileTotal *= i+1
            
        # TODO implement a scoring method to determine fitness

        return

class evolutionaryAlgorithm:
    population = []

    def __init__(self, populationSize =100, numberOfGenerations =100, genotypeLength =10):
        self.populationSize = populationSize
        self.genotypeLength = genotypeLength

    # Function that randomly generates a starting population
    def generatePopulation(self):
        for i in range(self.populationSize):
            individual = geneticIndividual()
            for j in range(self.genotypeLength):
                individual.genotype.append(random.randint(0,1))
            self.population.append(individual)
        return
            
    # Function to print the population
    def printPopulation(self):
        for individual in self.population:
            individual.printGeneticIndividual()
        return

    # TODO implement crossover function
    def crossover(self):
        return

    # Function to mutate a population, the mutateChance is in percentage
    # We mutate by swapping one value to the other pile
    def mutate(self, mutateChance =1):

        # For each individual in the population there is a chance equal to mutateChance to be mutated
        for individual in self.population:
            if random.randint(1, 100) <= mutateChance:
                mutateIndex = random.randint(0, self.genotypeLength-1)
                individual.genotype[mutateIndex] ^= 1

        return

algoritme = evolutionaryAlgorithm()
algoritme.generatePopulation()
algoritme.mutate()
