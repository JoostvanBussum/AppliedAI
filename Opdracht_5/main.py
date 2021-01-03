# our genotype will be a list that looks as follows(This is an example): [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] 
# Every position in the list represents one of the numbers(it being equal to index+1) and the 0 or 1 represents in what pile the number is allocated to(0 being the sum pile and 1 being the multiply pile)
# This way we can easily perform crossovers and mutations on the genotypes

import numpy
import copy
import math
import random

class geneticIndividual:
    def __init__(self, genotypeLength =10, genotype =None):
        self.fitness = 0

        # If genotype is None we initialise the genotype randomly else we assign it the given genotype
        if not genotype:
            self.genotype = [random.randint(0,1) for i in range(genotypeLength)]
            self.genotypeLength = genotypeLength
        else:
            self.genotype = genotype
            self.genotypeLength = len(genotype)

        # Evaluate fitness
        self.evaluateFitness()

    
    # Function to print genetic individuals
    def printGeneticIndividual(self):
        print("Fitness: ", self.fitness, " genotype: ", self.genotype)


    # Function to determine the fitness of an individual
    def evaluateFitness(self):
        sumPileTotal = 0

        # Count how many items are in pile 1, because if it is 1 or more the value needs to start at 1 else we multiply by 0
        if self.genotype.count(1) >= 1:
            productPileTotal = 1
        else: 
            productPileTotal = 0

        # Calculate the total value of the piles
        # Because i starts at 0 and every items value is equal to it's position we do i+1
        # so that item 1 (at index 0) still has a value of 1 and so on for all the items
        for i in range(len(self.genotype)):
            if self.genotype[i] == 0:
                sumPileTotal += i+1
            else:
                productPileTotal *= i+1
        
        # TODO implement a scoring method to determine fitness according to scaled error method
        self.fitness = (abs((sumPileTotal/36)/36) + abs((productPileTotal/360)/360)) * 100

        
class evolutionaryAlgorithm:
    population = []

    def __init__(self, populationSize =1000, numberOfGenerations =100, genotypeLength =10, mutateChance =0.01, retainModifier =0.2):
        self.populationSize = populationSize
        self.genotypeLength = genotypeLength
        self.mutateChance = mutateChance
        self.retainModifier = retainModifier
        self.retainLenght = abs(int(self.populationSize*retainModifier))

    # Function that randomly generates a starting population
    def generatePopulation(self):
        for i in range(self.populationSize):
            individual = geneticIndividual()
            self.population.append(individual)
        return
            
    # Function to print the population
    def printPopulation(self):
        graded = sorted(self.population, key=lambda x: x.fitness)
        for individual in graded:
            individual.printGeneticIndividual()
        return
    
    # TODO implement evolve functie
    def evolve(self):
        # Sort the individuals by fitness in a new list
        graded = sorted(self.population, key=lambda x: x.fitness)
        retainedParents = graded[:self.retainLenght] #TODO fitness ff fixen (fitness 2800 is nu 'beste' individual)
            
        # Function or class parameter?
        random_chance = 0.5
        for individual in graded[self.retainLenght:]:
            if random_chance >= random.uniform(0, 100):
                retainedParents.append(individual)

        desiredPopulationSize = self.populationSize - len(retainedParents)
        children = []

        return
        
        
    # TODO implement crossover function
    def crossover(self):
        
        return

    # Function to mutate a population, the mutateChance is in percentage
    # We mutate by swapping one value to the other pile
    def mutate(self):

        # For each gene for each individual in the population there is a chance equal to mutateChance to be mutated
        for individual in self.population:
            for gene in range(len(individual.genotype)):
                if random.uniform(0, 100) <= self.mutateChance:
                    individual.genotype[gene] ^= 1

        return

algoritme = evolutionaryAlgorithm()
algoritme.generatePopulation()
algoritme.printPopulation()