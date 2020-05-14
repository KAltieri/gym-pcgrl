import numpy
import geneticAlgorithmSupport

# A population of networks that are flattened weights
# Do the neuroevolution strategy outside of the class.

# Second approach, natural evolution: https://arxiv.org/pdf/1802.08842.pdf
# Show the best neural network after running for next week + results

# Parameters are: equation_inputs, num_weights, sol_per_pop, low, high, num_generations, num_parents_mating
def geneticAlgorithmMain(equation_inputs, num_weights, sol_per_pop, lowIn, highIn, num_generations, num_parents_mating, reward):
    # Initial population will be defined based on the number of weights, each chromosome
    # In the population will definitely have X genes, one for each weight.

    # Defining the population size as a tuple
    pop_size = (sol_per_pop, num_weights)

    # Now we create it
    new_population = numpy.random.uniform(low=lowIn, high=highIn, size=pop_size)

    # Select the best individuals within our current population as parents for "mating"
    # This'll basically be our fitness function, all done within a for loop

    # Within the same for loop, apply the GA variants (crossover + mutation) to produce the offspring of the next
    # generation This will, in turn, create a NEW population by appending both parents + offspring We repeat the
    # steps for a number of GENERATIONS (that's why the for loop is there)

    for generation in range(num_generations):
        #fitness = geneticAlgorithmSupport.cal_pop_fitness(equation_inputs, new_population)
        fitness = reward

        # First start off by measuring the fitness of each chromosome in the new population
        # Then we need to select the best parents for mating within this population
        parents = geneticAlgorithmSupport.select_mating_pool(new_population, fitness, num_parents_mating)

        # Note: The higher the fitness value, the better the solution!

        # Next generate the next generation via crossover!
        offspring_crossover = geneticAlgorithmSupport.crossover(parents, offspring_size=(pop_size[0] - parents.shape[0],
                                                                                         num_weights))

        # Mutations
        offspring_mutation = geneticAlgorithmSupport.mutation(offspring_crossover)

        # Finalizing the new population!
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    return new_population



"""
if __name__ == '__main__':
    # Below are test variables
    # Inputs of the equation.

    # Row (1) = run of gym-pcgrl
    # Column = step of that run
    # Cell = reward per step of specific gym-pcgrl run


    equation_inputs_test = [4, -2, 3.5, 5, -11, -4.7]  # This can be formulated by another script, one that takes in
    # inputs as the completed levels
    # Number of the weights we are looking to optimize.
    num_weights_test = 6  # This is going to change based on the weight we want to define for gym-PCGRL

    # The following variable will hold the number of solutions per population
    sol_per_pop_test = 8

    lowIn_test = -4.0
    highIn_test = 4.0

    num_generations_test = 5  # This can be changed
    num_parents_mating_test = 4  # As can this

    geneticAlgorithmMain(equation_inputs_test, num_weights_test, sol_per_pop_test, lowIn_test, highIn_test, num_generations_test,
         num_parents_mating_test)
"""