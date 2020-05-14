import numpy
import random


def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop * equation_inputs, axis=1)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next
    # generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999  # or MIN_INT
    return parents


# This function accepts the parents and the offspring size. It uses the offspring size to know the number of offspring
# To produce from such parents
def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)

    """
    Old crossover setup

    # The point at which crossover takes place between two parents. Usually it is at the center.
    # crossover_point = numpy.uint8(offspring_size[1] / 2)

    # Take half of the offspring_size

    # Parents are selected in a way similar to a ring. The first with indices 0 and 1 mate
    # Then we select parent 1 with parent 2 for another offspring, then 2 and 3, etc.
    # If we reach the last parent, then we mate parent last with parent 0

    """

    # Take n = offspring_size[0] amount of the parents from the mating pool!
    # Copy them and mutate them!
    random_samples = random.sample(range(0, parents.shape[0]), offspring_size[0])

    for k in range(offspring_size[0]):

        # Copying
        offspring[k, :] = parents[random_samples[k], :]
        """

        Old crossover implementation

        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]


        """

    # The next step will force all the offsprings (AKA parent copies) to go and be mutate at a index of each gene
    return offspring

# Instead copy one of the parents, and mutate it (it could be best or rank selection)

# This is known as a random-based optimization technique. It tries to enhance the current solutions
# by applying some random changes to them. Because such changes are random, we are not sure that they will produce
# better solution! Because of this, it is preferred to keep previous best solutions, aka the parents, in the new pop!
def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = numpy.random.normal(0, 0.1, 1) # Gaussian distribution with mean 0 and standard deviation of 0.1

        # Below affects all offsprings but only at a specified point
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value # Random # is then added to the gene with index 4 of the offspring according to this rule

    return offspring_crossover


