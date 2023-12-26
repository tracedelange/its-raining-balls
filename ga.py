from copy import deepcopy
import torch
from config import Config
from brain import Brain
import random
import matplotlib.pyplot as plt
import numpy as np
import time
from game import loop_game

con = Config()

# random.seed(0)


def mutate(agent, mutation_rate, mutation_strength):
    new_agent = deepcopy(agent)  # Create a copy of the winning agent
    for layer in new_agent.children():
        if torch.rand(()) < mutation_rate:
            for param in layer.parameters():
                param.data += torch.randn(param.shape) * mutation_strength
    return new_agent


def crossover(agent1, agent2):
    new_agent = deepcopy(agent1)  # Create a copy of one of the parents
    for layer1, layer2 in zip(new_agent.children(), agent2.children()):
        for param1, param2 in zip(layer1.parameters(), layer2.parameters()):
            alpha = torch.rand(())  # Weight for blending
            param1.data = alpha * param1.data + (1 - alpha) * param2.data
    return new_agent


def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        agent = Brain()
        population.append(agent)
    return population


def evaluate_fitness(population, gen=0):
    fitness_scores = []
    best_score = 0
    best_agent = None
    best_agent_seed = None
    # for agent in population:
    for i in range(len(population)):
        if con.same_seed:
            random_seed = 0
        else:
            random_seed = random.randint(0, 1000000000000000)
        score = loop_game(population[i], gen, i,
                          virtual=True, random_seed=random_seed)
        fitness_scores.append(score)
        if score > best_score:
            best_score = score
            best_agent = population[i]
            best_agent_seed = random_seed
    return fitness_scores, best_score, best_agent, best_agent_seed


def select_parents(population, fitness_scores):
    parents = []
    for _ in range(con.num_parents):
        # Randomly select k individuals from the population
        candidates = random.choices(population, k=con.tournament_size)
        # Select the best individual from the k individuals
        best_candidate = max(
            candidates, key=lambda agent: fitness_scores[population.index(agent)])
        parents.append(best_candidate)

    return parents


def crossover_and_mutate(parents):
    offspring = []
    for _ in range(con.population_size):
        # Randomly select two parents
        parent1, parent2 = random.sample(parents, k=2)
        # Crossover
        if torch.rand((1,)) < con.crossover_rate:
            child = crossover(parent1, parent2)
        else:
            child = deepcopy(parent1)
        # Mutate
        child = mutate(child, con.mutation_rate, con.mutation_strength)
        offspring.append(child)
    return offspring


def apply_elitism(offspring, population, fitness_scores):
    if con.elitism:
        # Select the best individuals from the old population
        elites = sorted(population, key=lambda agent: fitness_scores[population.index(
            agent)], reverse=True)[:con.num_elites]
        # Replace the worst individuals from the new population with the elites
        offspring[-con.num_elites:] = elites
    return offspring


def get_best_individual(population):
    return max(population, key=lambda agent: evaluate_network(agent))


def generate_training_graph(output):
    # Graph output and save it as a png
    x = []
    y = []
    for item in output:
        x.append(item[0])
        y.append(item[1])

    # Fit a linear regression to score
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
    else:
        m = 0
        b = 0

    # Calculate the y-values for the linear regression line
    regression_line = m * np.array(x) + b

    # Clear the plot
    plt.clf()

    # Plot the data
    plt.plot(x, y, label='Data')

    # Plot the linear regression line
    plt.plot(x, regression_line, label='Linear Regression Line', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Training Progress')
    # // TODO: Ensure path exists
    plt.savefig('./figures/training_graph.png')


def train():

    # Initialize population
    population = initialize_population(con.population_size)
    print('Population initialized...')

    output = []

    # Main GA loop
    for generation in range(con.num_generations):

        start_time = time.time()
        # Evaluate fitness of individuals
        fitness_scores, best_score, best_agent, best_agent_seed = evaluate_fitness(
            population, generation)

        # Select parents
        parents = select_parents(population, fitness_scores)

        # Create offspring through crossover and mutation
        offspring = crossover_and_mutate(parents)

        # Apply elitism
        offspring = apply_elitism(offspring, population, fitness_scores)

        # Replace the old population with the new population
        population = offspring

        print('Generation', generation, 'complete...')
        print('highest score for generation ' + str(generation) + ': ' +
              str(best_score) + ", time: " + str(time.time() - start_time))
        # Save best agent:

        torch.save(best_agent.state_dict(), './models/' +
                   str(generation) + "_" + str(best_agent_seed) + '.pt')

        output.append((generation, best_score))

        generate_training_graph(output)

        # render every x generation
        if generation % con.preview_mod == 0 and con.preview_active:
            loop_game(best_agent, generation, individual_index=None, virtual=False,
                      random_seed=best_agent_seed)

    # Termination condition reached
    best_individual = get_best_individual(population)
    print('Best individual:', best_individual)

    # torch.save(best_individual.state_dict(), './models/debug.pt')

    # write output to file
    with open('./output.txt', 'w') as f:
        for item in output:
            f.write("%s\n" % str(item))
