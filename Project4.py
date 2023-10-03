import random
import math

max_generations = 100  # Adjust as needed
pop_size = 100  # Adjust as needed
mutation_rate = 0.2  # Adjust as needed


class Individual:
    def __init__(self, city_ids):
        self.route = random.sample(city_ids, len(city_ids))
        self.fitness = None

    def swap_mutation(self):
        # Select two random indices
        idx1, idx2 = random.sample(range(len(self.route)), 2)
        # Swap the cities at the selected indices
        self.route[idx1], self.route[idx2] = self.route[idx2], self.route[idx1]
        self.fitness = None  # Reset fitness to be recalculated

    def scramble_mutation(self):
        # Select a random subset of cities
        start_idx, end_idx = sorted(random.sample(range(len(self.route)), 2))
        # Shuffle the subset of cities
        subset = self.route[start_idx:end_idx + 1]
        random.shuffle(subset)
        # Replace the shuffled subset in the route
        self.route[start_idx:end_idx + 1] = subset
        self.fitness = None  # Reset fitness to be recalculated

class Population:
    def __init__(self, size, city_ids):
        self.individuals = [Individual(city_ids) for _ in range(size)]

    def uniform_crossover(self, parent1, parent2):
        child1 = [-1] * len(parent1.route)
        child2 = [-1] * len(parent2.route)

        print("Parent 1: ", parent1.route)  # Print Parent 1 route
        print("Parent 2: ", parent2.route)  # Print Parent 2 route

        for i in range(len(parent1.route)):
            if random.random() < 0.5:
                child1[i] = parent1.route[i]
                child2[i] = parent2.route[i]
            else:
                child1[i] = parent2.route[i]
                child2[i] = parent1.route[i]
        
        print("Child 1 before fix:", child1)  # Debugging print
        print("Child 2 before fix:", child2)  # Debugging print

        # Ensure that each child is a valid route
        child1 = self.fix_invalid_route(child1)
        child2 = self.fix_invalid_route(child2)

        return child1, child2

    def cycle_crossover(self, parent1, parent2):
        cycle = [-1] * len(parent1.route)
        start_idx = 0


        print("Parent 1: ", parent1.route)  # Print Parent 1 route
        print("Parent 2: ", parent2.route)  # Print Parent 2 route


        while -1 in cycle:
            if cycle[start_idx] == -1:
                cycle[start_idx] = parent1.route[start_idx]
                next_idx = parent2.route.index(cycle[start_idx])
                while next_idx != start_idx:
                    cycle[next_idx] = parent1.route[next_idx]
                    next_idx = parent2.route.index(cycle[next_idx])
            start_idx = cycle.index(-1)

        # Create offspring routes based on the cycle
        child1 = [-1 if city not in cycle else city for city in parent1.route]
        child2 = [-1 if city not in cycle else city for city in parent2.route]

        # Ensure that each child is a valid route
        child1 = self.fix_invalid_route(child1)
        child2 = self.fix_invalid_route(child2)

        return child1, child2
    
    def fix_invalid_route(self, route):
        # Ensure that each city appears exactly once in the route
        missing_cities = set(range(1, len(route) + 1)) - set(route)
        for city in missing_cities:
            if -1 in route:
                idx = route.index(-1)
                route[idx] = city
        return route
    

def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def initialize_population(pop_size, city_ids):
    return Population(pop_size, city_ids)

def calculate_fitness(individual, city_coordinates):
    total_distance = 0.0
    for i in range(len(individual.route) - 1):
        city1 = city_coordinates[individual.route[i]]
        city2 = city_coordinates[individual.route[i + 1]]
        total_distance += calculate_distance(city1, city2)
    # Add the distance from the last city to the first city to complete the cycle
    total_distance += calculate_distance(city_coordinates[individual.route[-1]], city_coordinates[individual.route[0]])
    individual.fitness = total_distance
    return total_distance


def roulette_wheel_selection(population):
    total_fitness = sum(1 / ind.fitness for ind in population.individuals)
    selected = random.uniform(0, total_fitness)
    current_sum = 0
    for ind in population.individuals:
        current_sum += 1 / ind.fitness
        if current_sum >= selected:
            return ind
        
def check_termination_criteria(population, generation, max_generations, fitness_threshold):
    # Check if the maximum number of generations has been reached
    if generation >= max_generations:
        return True
    
    valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]
    
    if not valid_individuals:
        return True
    

    # Alternatively, you can check if the best fitness meets a certain threshold
    best_fitness = min(population.individuals, key=lambda x: x.fitness).fitness
    if best_fitness <= fitness_threshold:
        return True
    
    return False  # Continue running the algorithm

# Initialize your variables (e.g., max_generations, fitness_threshold)
max_generations = 1000  # Replace with your desired maximum number of generations
fitness_threshold = 1000.0  # Replace with your desired fitness threshold


# Initialize variables to store data
city_data = {}
dimension = 0
read_coordinates = False  # Flag to indicate when to start reading coordinates

# Open the file for reading
file_path = 'Random100.tsp'  # Replace with the actual file path
with open(file_path, 'r') as file:
    for line in file:
        # Split each line by whitespace and remove leading/trailing spaces
        parts = line.strip().split()

        # Check for specific keywords and extract data
        if parts[0] == "DIMENSION:":
            dimension = int(parts[1])
        elif parts[0] == "NODE_COORD_SECTION":
            read_coordinates = True
        elif read_coordinates and len(parts) == 3:
            # Assuming the format is "City_ID X_Coordinate Y_Coordinate"
            city_id = int(parts[0])
            x_coordinate = float(parts[1])
            y_coordinate = float(parts[2])
            city_data[city_id] = (x_coordinate, y_coordinate)

# Now, 'dimension' contains the number of cities, and 'city_data' is a dictionary
# where keys are city IDs and values are tuples of (X, Y) coordinates.

# You can access and print the data like this:
print(f"Number of cities: {dimension}")
for city_id, coordinates in city_data.items():
    x, y = coordinates
    print(f"City {city_id}: X={x}, Y={y}")

city_ids = list(range(1, dimension + 1))


population = initialize_population(pop_size, city_ids)

for generation in range(max_generations):
    # Evaluate fitness for each individual in the population
    for individual in population.individuals:
        calculate_fitness(individual, city_data)

    # Select parents for crossover (e.g., using roulette wheel selection)
    parents = [roulette_wheel_selection(population) for _ in range(pop_size)]

    # Create a new population through crossover (e.g., using Uniform or Cycle Crossover)
    new_population = Population(pop_size, city_ids)
    for i in range(0, pop_size, 2):
        child1, child2 = population.uniform_crossover(parents[i], parents[i + 1])
        new_population.individuals[i] = Individual(child1)  # Create Individual objects for children
        new_population.individuals[i + 1] = Individual(child2)  
    # Apply mutation operators (e.g., Swap Mutation or Scramble Mutation)
    for new_individual in new_population.individuals:
        if random.random() < mutation_rate:
            new_individual.swap_mutation()  # Apply Swap Mutation with some probability
        if random.random() < mutation_rate:
            new_individual.scramble_mutation()  # Apply Scramble Mutation with some probability

    # Replace the old population with the new population
    population = new_population

    # Termination criteria (e.g., stop if a certain fitness threshold is reached)
    if check_termination_criteria(population, generation, max_generations, fitness_threshold):
        break


# Calculate fitness for all individuals in the population
for individual in population.individuals:
    calculate_fitness(individual, city_data)

# Filter out individuals with None fitness values and find the best among the rest
valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]

if valid_individuals:
    best_solution = min(valid_individuals, key=lambda x: x.fitness)
    print(f"Best solution found: {best_solution.route}")
    print(f"Total distance: {best_solution.fitness}")
else:
    print("No valid solutions found.")



# valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]
#best_solution = min(population.individuals, key=lambda x: x.fitness)
#print(f"Best solution found: {best_solution.route}")
#print(f"Total distance: {best_solution.fitness}")

# Parameter Experimentation Loop
# Parameter Experimentation Loop

#crossover_methods = ["Uniform", "Cycle"]
#mutation_methods = ["Swap", "Scramble"]

crossover_methods = ["Cycle"]
mutation_methods = ["Swap"]

for crossover_method in crossover_methods:
    for mutation_method in mutation_methods:
        if crossover_method == "Uniform" and mutation_method == "Swap":
            pop_size = 200
            mutation_rate = 0.1
        elif crossover_method == "Cycle" and mutation_method == "Scramble":
            pop_size = 150
            mutation_rate = 0.2
        
        # Run the GA loop as shown earlier in your code
        population = initialize_population(pop_size, city_ids)

        for generation in range(max_generations):
            for individual in population.individuals:
                calculate_fitness(individual, city_data)

            # Select parents for crossover (e.g., using roulette wheel selection)
            parents = [roulette_wheel_selection(population) for _ in range(pop_size)]

            # Create a new population through crossover (e.g., using Uniform or Cycle Crossover)
            new_population = Population(pop_size, city_ids)
            for i in range(0, pop_size, 2):
                child1, child2 = population.uniform_crossover(parents[i], parents[i + 1])
                new_population.individuals[i] = Individual(child1)
                new_population.individuals[i + 1] = Individual(child2)

            # Apply mutation operators (e.g., Swap Mutation or Scramble Mutation)
            for individual in new_population.individuals:
                if random.random() < mutation_rate:
                    individual.swap_mutation()  # Apply Swap Mutation with some probability
                if random.random() < mutation_rate:
                    print(type(individual))
                    individual.scramble_mutation()  # Apply Scramble Mutation with some probability

            # Replace the old population with the new population
            population = new_population

            # Termination criteria (e.g., stop if a certain fitness threshold is reached)
            if check_termination_criteria(population, generation, max_generations, fitness_threshold):
                break


        for individual in population.individuals:
            calculate_fitness(individual, city_data)

        valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]

        if valid_individuals:
            best_solution = min(valid_individuals, key=lambda x: x.fitness)
            print(f"Results for Crossover: {crossover_method}, Mutation: {mutation_method}")
            print(f"Best solution found: {best_solution.route}")
            print(f"Total distance: {best_solution.fitness}")
        else:
            print("No valid solutions found.")



        #best_solution = min(population.individuals, key=lambda x: x.fitness)
        #print(f"Results for Crossover: {crossover_method}, Mutation: {mutation_method}")
        #print(f"Best solution found: {best_solution.route}")
        #print(f"Total distance: {best_solution.fitness}")
