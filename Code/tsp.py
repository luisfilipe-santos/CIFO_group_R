from charles.charles import Population, Individual
from data.tsp_data import distance_matrix, distance_matrix_bavaria, distance_matrix_bangladesh
from copy import deepcopy
from charles.selection import fps, tournament, ranking
from charles.mutation import swap_mutation, inversion_mutation, scramble_mutation
from charles.crossover import cycle_co, pmx_co, OBX

import time
import pandas as pd
import numpy as np
import plotly.express as px


def get_fitness(self):
    """A simple objective function to calculate distances
    for the TSP problem.

    Returns:
        int: the total distance of the path
    """
    fitness = 0
    for i in range(len(self.representation)):
        fitness += distance_matrix_bavaria[self.representation[i - 1]][self.representation[i]]
    return int(fitness)


def get_neighbours(self):
    """A neighbourhood function for the TSP problem. Switches
    indexes around in pairs.

    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation) - 1)]

    for count, i in enumerate(n):
        i[count], i[count + 1] = i[count + 1], i[count]

    n = [Individual(i) for i in n]
    return n


# Monkey patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours



# From the charles lib
# Selection methods used: fps, tournament, ranking
# Crossover methods used: cycle_co, pmx_co, OBX
# Mutation methods used: swap_mutation, inversion_mutation, scramble_mutation

selection_used = [fps, tournament, ranking]
crossover_used = [cycle_co, pmx_co, OBX]
mutation_used = [swap_mutation, inversion_mutation, scramble_mutation]

# Combination Names
names = ["fps_cycle_swap", "fps_cycle_inversion", "fps_cycle_scramble", "fps_pmx_swap", "fps_pmx_inversion", "fps_pmx_scramble", "fps_OBX_swap", "fps_OBX_inversion", "fps_OBX_scramble",
         "tournament_cycle_swap", "tournament_cycle_inversion", "tournament_cycle_scramble", "tournament_pmx_swap", "tournament_pmx_inversion", "tournament_pmx_scramble", "tournament_OBX_swap", "tournament_OBX_inversion", "tournament_OBX_scramble",
         "ranking_cycle_swap", "ranking_cycle_inversion", "ranking_cycle_scramble", "ranking_pmx_swap", "ranking_pmx_inversion", "ranking_pmx_scramble", "ranking_OBX_swap", "ranking_OBX_inversion", "ranking_OBX_scramble"]

# after benchmarking, this were the best combinations for each dataset
bav_best_name = ['tournament_OBX_inversion', 'tournament_pmx_inversion', 'ranking_OBX_inversion']
bang_best_name = ['fps_OBX_swap', 'fps_OBX_scramble', 'tournament_cycle_swap']
default_best_name = ['ranking_OBX_inversion ', 'tournament_OBX_inversion', 'tournament_cycle_inversion ']

fitness_per_combination = []
times_per_combination = []
final_df = pd.DataFrame()

count = 0
for selection in selection_used:
    for crossover in crossover_used:
        for mutation in mutation_used:
            avg_final_fit = 0
            fitness_progression_avged = []
            time_avged = 0
            for i in range(100):        # Amount of runs per combination
                start = time.time()     # Time per Run
                pop = Population(
                    size=50,
                    sol_size=len(distance_matrix_bavaria[0]),
                    valid_set=[i for i in range(len(distance_matrix_bavaria[0]))],
                    replacement=False,
                    optim="min",
                )

                pop.evolve(
                    gens=100,
                    select=selection,
                    crossover=crossover,
                    mutate=mutation,
                    co_p=0.9,
                    mu_p=0.05,
                    elitism=True
                )

                end = time.time()
                time_avged += (end-start)

                avg_final_fit += pop.fit_display[-1]
                fitness_progression_avged.append(pop.fit_display)

            time_avged = time_avged / 100
            times_per_combination.append(time_avged)

            # Average final fitness of all runs
            avg_final_fit = avg_final_fit / 100
            fitness_per_combination.append(avg_final_fit)

            # Average fitness progression of each generation of all runs
            numpy_array = np.array(fitness_progression_avged)
            transpose = numpy_array.T

            df = pd.DataFrame(transpose)

            final_df[names[count]] = df.mean(axis=1)

            count += 1

times_df = pd.DataFrame(times_per_combination)
fitness_df = pd.DataFrame(fitness_per_combination)
times_df.index = names
fitness_df.index = names

print()
print(times_df)
print()
print(fitness_df)


#VISUALIZATIONS
fig = px.line(final_df, labels={
                     "index": "Generations",
                     "value": "Fitness",
                     "variable": "Combinations"
                    },)
fig.show()

fig = px.bar(times_df, orientation='h', labels={
                     "index": "Combinations",
                     "value": "Time (sec)",
                     "variable": "Combinations"
                    })
fig.update_xaxes(range=[0.0, 0.7])
fig.update_layout(showlegend=False)
fig.show()

fig = px.bar(fitness_df, orientation='h', labels={
                     "index": "Combinations",
                     "value": "Fitness",
                     "variable": "Combinations"
                    })
fig.update_xaxes(range=[2400, 4300])
fig.update_layout(showlegend=False)
fig.show()