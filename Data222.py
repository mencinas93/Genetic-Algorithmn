import random
import math
import tkinter as tk
import matplotlib.pyplot as plt
import time


max_generations = 100
pop_size = 100
mutation_rate = 0.2


class Individual:
    def __init__(self, city_ids):
        #initialize an individual with random route that includes all city ids. 
        #making sure route starts and ends at same city id
        self.route = random.sample(city_ids, len(city_ids))
        self.route.append(self.route[0])
        self.fitness = None

    def swap_mutation(self):
        # Selects two random cities in the route and swaps their positions.
        index1, index2 = random.sample(range(1, len(self.route) - 1), 2)
        self.route[index1], self.route[index2] = self.route[index2], self.route[index1]
        self.fitness = None
        # Reset fitness to be recalculated
        # Select two random indices within the route, excluding the first and last city.

    def scramble_mutation(self):
        # It selects a random subset of cities and shuffles their order.
        start_index, end_index = sorted(random.sample(range(1, len(self.route) - 1), 2))
        # Shuffles the subset of cities
        subset = self.route[start_index:end_index + 1]
        random.shuffle(subset)
        # Replaces the shuffled subset in the route
        self.route[start_index:end_index + 1] = subset
        self.fitness = None


class Population:
    def __init__(self, size, city_ids):
        # Initialize a population with 'size' individuals, each wih a random route.
        self.individuals = [Individual(city_ids) for _ in range(size)]

    def uniform_crossover(self, parent1, parent2):
        # Uniform Crossover: Combines two parent routes to create two child routes.
        child1 = [-1] * len(parent1.route)
        child2 = [-1] * len(parent2.route)
        # Initialize child routes as lists of -1s.

        # Iterate through each city in the parent routes.
        for i in range(len(parent1.route)):
             # Randomly select a parent to inherit the city from (50% chance each).
            if random.random() < 0.5:
                child1[i] = parent1.route[i]
                child2[i] = parent2.route[i]
            else:
                child1[i] = parent2.route[i]
                child2[i] = parent1.route[i]
        #Fix an invalid routes for complete tour
        child1 = self.fix_invalid_route(child1)
        child2 = self.fix_invalid_route(child2)

        return child1, child2

    def cycle_crossover(self, parent1, parent2):
        # Combines two parent routes using the cycle crossover method.
        cycle = [-1] * len(parent1.route)
        start_idx = 0
        # Initialize a list to represent the cycle and starting index
        while -1 in cycle:
            if cycle[start_index] == -1:
                cycle[start_index] = parent1.route[start_idx]
                next_index = parent2.route.index(cycle[start_idx])
                while next_index != start_idx:
                    cycle[next_index] = parent1.route[next_index]
                    next_index = parent2.route.index(cycle[next_index])
            start_index = cycle.index(-1)
        # keep repeating until all elements in the cycle are filled. 
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
                index = route.index(-1)
                route[index] = city
        return route

# Euclid formula - calculates distance between two cities. 
def calculated_distance_cities(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def initialize_population(pop_total_size, city_ids):
    return Population(pop_total_size, city_ids)


def calculate_fitness(individual, city_coordinates):
    total_traveled_distance = 0.0
    #calculates the total distance traveled for route/path
    for i in range(len(individual.route) - 1):
        city1 = city_coordinates[individual.route[i]]
        city2 = city_coordinates[individual.route[i + 1]]
        total_traveled_distance += calculated_distance_cities(city1, city2)
    # Add the distance from the last city to the first city to complete the cycle
    total_traveled_distance += calculated_distance_cities(city_coordinates[individual.route[-1]], city_coordinates[individual.route[0]])
    individual.fitness = total_traveled_distance
    return total_traveled_distance
    # Path/tour distance traveled. 

    # Perform roulette wheel selection to choose an individual from the population
def roulette_wheel_selection(population):
    total_fitness = sum(1 / ind.fitness for ind in population.individuals)
    random_selection= random.uniform(0, total_fitness)
    accumaleted_fitness = 0
    for ind in population.individuals:
        accumaleted_fitness += 1 / ind.fitness
        if accumaleted_fitness>= random_selection:
            return ind


def check_termination_criteria(population, generation, max_generations, fitness_threshold):
    # To check if the maximum number of generations has been reached
    if generation >= max_generations:
        return True
    # Filtering out individuals with invalid fitness values
    valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]

    if not valid_individuals:
        return True

    # Checking if the best fitness meets a certain threshold
    best_fitness = min(population.individuals, key=lambda x: x.fitness).fitness
    if best_fitness <= fitness_threshold:
        return True

    return False



max_generations = 1000  # Desired maximum number of generations
fitness_threshold = 1000.0  # Desired fitness threshold

# Initialize variables to store data
city_data = {}
dimension = 0 # number of cities
read_coordinates = False  # Flag to indicate when to start reading coordinates

# Open the file for reading
file_path = 'Random100.tsp'  # Replace with the actual file path
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()

        # Check for specific keywords to extract file data
        if parts[0] == "DIMENSION:":
            dimension = int(parts[1])
        elif parts[0] == "NODE_COORD_SECTION":
            read_coordinates = True
        elif read_coordinates and len(parts) == 3:
            city_id = int(parts[0])
            x_coordinate = float(parts[1])
            y_coordinate = float(parts[2])
            city_data[city_id] = (x_coordinate, y_coordinate)

city_ids = list(range(1, dimension + 1))


class TSPGUI:
    def __init__(self, root, city_data, crossover_method, mutation_method):
        self.root = root
        self.city_data = city_data
        self.best_solution = None  # Stores best solution
        self.city_labels = {}  # Initialize the city_labels dictionary
        self.scale_factor = 4  # Scaling factor
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        self.canvas = tk.Canvas(root, width=600, height=600)  # Increase the width and height
        self.canvas.pack() # Canvas

        self.start_button = tk.Button(root, text="Click for starting GA", command=self.genetic_algorithm)
        self.start_button.pack()

        self.best_solution_label = tk.Label(root, text="Best Solution: ")
        self.best_solution_label.pack()

        self.method_label = tk.Label(root, text=f"Crossover: {crossover_method}, Mutation: {mutation_method}")
        self.method_label.pack()

    def drawing_route(self, route):
        # Clear the canvas
        self.canvas.delete("all")

        for i in range(len(route) - 1):
            city1 = self.city_data[route[i]]
            city2 = self.city_data[route[i + 1]]
            x1, y1 = city1
            x2, y2 = city2
            x1_scaled, y1_scaled = x1 * self.scale_factor, y1 * self.scale_factor
            x2_scaled, y2_scaled = x2 * self.scale_factor, y2 * self.scale_factor
            self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="green", width=1)
        # Draw a line connecting the last city to the first to complete the cycle
        city1 = self.city_data[route[-1]]
        city2 = self.city_data[route[0]]
        x1, y1 = city1
        x2, y2 = city2
        x1_scaled, y1_scaled = x1 * self.scale_factor, y1 * self.scale_factor
        x2_scaled, y2_scaled = x2 * self.scale_factor, y2 * self.scale_factor
        self.canvas.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="green", width=5)
        # Labels
        for city_id, coordinates in self.city_data.items():
            x, y = coordinates
            x_scaled, y_scaled = x * self.scale_factor, y * self.scale_factor
            label = self.city_labels.get(city_id)
            if label:
                self.canvas.delete(label)  # Remove previous label
            label = self.canvas.create_text(x_scaled, y_scaled, text=str(city_id), font=("Arial", 12, "bold"), fill="red")
            self.city_labels[city_id] = label

    def update_gui(self):
        #Updating GUI with best solution's route and fitness
        if self.best_solution:
            self.drawing_route(self.best_solution.route)
            self.best_solution_label.config(text=f"Best Solution GA tour: {self.best_solution.fitness:.2f}")

    def genetic_algorithm(self):
        global max_generations, fitness_threshold, pop_size, mutation_rate
         # Initializes the population
        population = initialize_population(pop_size, city_ids)

        for generation in range(max_generations):
            # Calculates fitness for each individual
            for individual in population.individuals:
                calculate_fitness(individual, city_data)
            
            # Selecting parents using roulette wheel selection
            parents = [roulette_wheel_selection(population) for _ in range(pop_size)]

            new_population = Population(pop_size, city_ids)
            for i in range(0, pop_size, 2):
                child1, child2 = population.uniform_crossover(parents[i], parents[i + 1])
                new_population.individuals[i] = Individual(child1)
                new_population.individuals[i + 1] = Individual(child2)

            
            # Applying mutation to the new population
            for individual in new_population.individuals:
                if random.random() < mutation_rate:
                    individual.swap_mutation()
                if random.random() < mutation_rate:
                    individual.scramble_mutation()

            population = new_population

            if check_termination_criteria(population, generation, max_generations, fitness_threshold):
                break
        # Calculates fitness for all individuals in the final total population
        for individual in population.individuals:
            calculate_fitness(individual, city_data)
        # Filtering valid individuals and finding the best solution
        valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]

        if valid_individuals:
            self.best_solution = min(valid_individuals, key=lambda x: x.fitness)
            self.update_gui()  # Update the GUI with the best solution
        else:
            print("No valid solutions found!")


def main():
    root = tk.Tk()
    root.title("TSP GA")
    gui = TSPGUI(root, city_data, "Uniform", "Swap")
    root.mainloop()


if __name__ == "__main__":
    main()

# Crossover and mutation methods to be tested
crossover_methods = ["Cycle", "Uniform"]
mutation_methods = ["Swap", "Scramble"]

fitness_dict = {}

for crossover_method in crossover_methods:
    for mutation_method in mutation_methods:
        if crossover_method == "Uniform" and mutation_method == "Swap":
            pop_size = 200
            mutation_rate = 0.1
        elif crossover_method == "Cycle" and mutation_method == "Scramble":
            pop_size = 150
            mutation_rate = 0.2
        else:
            pop_size = 100
            mutation_rate = 0.2

        results = []
        start_time = time.time() 
        # Run the GA loop
        for _ in range(100):  # number of run times
            population = initialize_population(pop_size, city_ids)

            for generation in range(max_generations):
                for individual in population.individuals:
                    calculate_fitness(individual, city_data)

                parents = [roulette_wheel_selection(population) for _ in range(pop_size)]

                new_population = Population(pop_size, city_ids)
                for i in range(0, pop_size, 2):
                    child1, child2 = population.uniform_crossover(parents[i], parents[i + 1])
                    new_population.individuals[i] = Individual(child1)
                    new_population.individuals[i + 1] = Individual(child2)

                for individual in new_population.individuals:
                    if random.random() < mutation_rate:
                        individual.swap_mutation()
                    if random.random() < mutation_rate:
                        individual.scramble_mutation()

                population = new_population

                if check_termination_criteria(population, generation, max_generations, fitness_threshold):
                    break

            for individual in population.individuals:
                calculate_fitness(individual, city_data)

            valid_individuals = [ind for ind in population.individuals if ind.fitness is not None]

            if valid_individuals:
                best_solution = min(valid_individuals, key=lambda x: x.fitness)
                results.append(best_solution.fitness)
                print(f"Results for Crossover: {crossover_method}, Mutation is: {mutation_method}")
                print(f"Best solution found is: {best_solution.route}")
                print(f"Total distance traveled in tour: {best_solution.fitness}")
            else:
                print("No valid solutions found!")


        fitness_dict[(crossover_method, mutation_method)] = results
        print(f"Results for Crossover: {crossover_method}, Mutation is: {mutation_method}")


        if results:
            mean_fitness = sum(results) / len(results)
            min_fitness = min(results)
            max_fitness = max(results)
            std_deviation = math.sqrt(sum((x - mean_fitness) ** 2 for x in results) / len(results))

            print(f"Mean fitness for 100 runs: {mean_fitness:.2f}")
            print(f"Minimum fitness for 100 runs: {min_fitness:.2f}")
            print(f"Maximum fitness for 100 runs: {max_fitness:.2f}")
            print(f"Standard deviation for 100 runs: {std_deviation:.2f}")
        else:
            print("No valid solutions found!")

        end_time = time.time()  # Records the end time
        execution_time = end_time - start_time
        print(f"Execution time for 100 runs: {execution_time:.2f} seconds")

for combo, fitness_values in fitness_dict.items():
    crossover_method, mutation_method = combo
    generations = list(range(len(fitness_values)))
    
    if (crossover_method == "Cycle" and mutation_method == "Swap") or (crossover_method == "Uniform" and mutation_method == "Scramble"):
        label = f"{crossover_method}, {mutation_method}"
        
        if fitness_values:
            plt.plot(generations, fitness_values, label=label)

plt.xlabel("Generation")
plt.ylabel("Cost")
plt.legend(loc="best")
plt.title("Improvement Curves for two selected GA combinations")
plt.show()
Curves for Different GA Combinations")
#plt.show()
