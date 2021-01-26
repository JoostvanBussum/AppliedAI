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

    
    # Function to print the genetic individual
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
        perfectFitness = 100

        sumPileScore = abs((sumPileTotal/36 * 100) - perfectFitness)
        productPileScore = abs((productPileTotal/360 * 100) - perfectFitness)
        
        self.fitness = sumPileScore + productPileScore
        
class evolutionaryAlgorithm:

    def __init__(self, populationSize =100, numberOfGenerations =100, genotypeLength =10, mutateChance =50, retainModifier =0.2):
        self.populationSize = populationSize
        self.generations = numberOfGenerations
        self.genotypeLength = genotypeLength
        self.mutateChance = mutateChance
        self.retainModifier = retainModifier
        self.retainLenght = abs(int(self.populationSize*retainModifier))
        self.population = []

    # Function that randomly generates a starting population
    def generatePopulation(self):
        for i in range(self.populationSize):
            individual = geneticIndividual()
            self.population.append(individual)
            
    # Function to print the population
    def printPopulation(self):
        graded = sorted(self.population, key=lambda x: x.fitness, reverse=False)
        for individual in graded:
            individual.printGeneticIndividual()
    
    # Evolve function that first selects parents and then does crossover and mutate
    def evolve(self):
        # Sort the individuals by fitness in a new list
        graded = sorted(self.population, key=lambda x: x.fitness, reverse=False)
        retainedParents = graded[:self.retainLenght] 
        
        # Keep some less good parents for more diversity
        # Function or class parameter?
        random_chance = 0.5
        for individual in graded[self.retainLenght:]:
            if random_chance >= random.uniform(0, 100):
                retainedParents.append(individual)

        # Even out the list to a even number if necessary
        if len(retainedParents) % 2 == 1:
            retainedParents.append(graded[random.randint(0, self.retainLenght)])
            
        offspringAmount = self.populationSize - len(retainedParents)
        children = []
        while  len(children) < offspringAmount:

            # Make offspring

            offspringChildren = self.crossover(random.sample(retainedParents, 2))

            # Perform possible mutation on all the children
            children.append(self.mutate(offspringChildren[0]))
            children.append(self.mutate(offspringChildren[1]))


        retainedParents.extend(children)
        self.population = retainedParents.copy()
        
    # Single point crossover function with the point being random
    def crossover(self, parents):
        

        pointIndex = random.randint(1, parents[0].genotypeLength-1)


        child1Genotype = parents[0].genotype[:pointIndex]
        child1Genotype.extend(parents[1].genotype[pointIndex:])

        child2Genotype = parents[1].genotype[:pointIndex]
        child2Genotype.extend(parents[0].genotype[pointIndex:])

        children = [geneticIndividual(genotype = child1Genotype), geneticIndividual(genotype = child2Genotype)]
        return children

    # Function to mutate a population, the mutateChance is in percentage
    # We mutate by swapping one value to the other pile
    def mutate(self, individual):

        # For each gene for each individual in the population there is a chance equal to mutateChance to be mutated
        for gene in range(len(individual.genotype)):
            if random.uniform(0, 100) <= self.mutateChance:
                #individual.genotype[gene] != individual.genotype[gene]
                individual.genotype[gene] ^= 1
                individual.evaluateFitness()

        return individual

    # Function that runs the whole algorithm an amount of times equal to generations
    def run(self):
        for iteration in range(self.generations):
            self.evolve()

        return sorted(self.population, key=lambda x: x.fitness, reverse=False)


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