from random import uniform, choice
from operator import attrgetter


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":
        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        # Sum total fitness
        sum_fitness = sum([i.fitness for i in population])

        total_fitness = 0
        new_fitnesses = []
        for i in population:
            total_fitness += (sum_fitness - i.fitness)
            new_fitnesses.append(sum_fitness - i.fitness)

        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin

        count = 0
        for individual in population:
            position += new_fitnesses[count]
            if position > spin:
                return individual
            count += 1

    else:
        raise Exception("No optimization specified (min or max).")


def tournament(population, size=10):
    """Tournament selection implementation.

    Args:
        population (Population): The population we want to select from.
        size (int): Size of the tournament.

    Returns:
        Individual: Best individual in the tournament.
    """

    # Select individuals based on tournament size
    tournament = [choice(population.individuals) for i in range(size)]
    # Check if the problem is max or min
    if population.optim == 'max':
        return max(tournament, key=attrgetter("fitness"))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter("fitness"))
    else:
        raise Exception("No optimization specified (min or max).")

def ranking(population):
    """Ranking selection implementation.

        Args:
            population (Population): The population we want to select from.

        Returns:
            Individual: selected individual.
    """
    if population.optim == "max":
        rank = [individual for individual in population]

        # define a sort key
        def sort_key(individual):
            return individual.fitness

        # ranking the individuals
        rank.sort(key=sort_key)

        # These next steps are similar to the fps method
        total_weight = 0
        for i in range(len(rank)):
            total_weight += (i+1)
        # Get a 'position' on the wheel
        spin = uniform(0, total_weight)
        position = 0
        # Find individual in the position of the spin
        for count, individual in enumerate(rank):
            position += (count + 1)
            if position > spin:
                return individual

    if population.optim == "min":
        rank = [individual for individual in population]

        # define a sort key
        def sort_key(individual):
            return individual.fitness

        # ranking the individuals
        rank.sort(key=sort_key, reverse=True)

        # These next steps are similar to the fps method
        total_weight = 0
        for i in range(len(rank)):
            total_weight += (i + 1)
        # Get a 'position' on the wheel
        spin = uniform(0, total_weight)
        position = 0
        # Find individual in the position of the spin
        for count, individual in enumerate(rank):
            position += (count + 1)
            if position > spin:
                return individual

