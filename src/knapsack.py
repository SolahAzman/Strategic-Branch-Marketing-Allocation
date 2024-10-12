# pandas, numpy, matplotlib, seaborn
import pandas as pd
import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression

# Display Option
pd.set_option('display.max_rows', 500) 
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000000)
pd.options.display.float_format = '{:.2f}'.format

def knapsack_greedy_ratio(data, itm, val, wgh, capacity):
    df = data.copy()
    df['ratio'] = df[val] / df[wgh]
    sorted_df = df.sort_values('ratio', ascending=False)
    
    total_weight = total_value = 0
    selected_items = []
    
    for _, item in sorted_df.iterrows():
        if total_weight + item[wgh] <= capacity:
            selected_items.append(item[itm])
            total_weight += item[wgh]
            total_value += item[val]
    
    return pd.DataFrame({
                        'Technique' : 'Greedy Ratio',
                        'Items_Name': [selected_items],
                        'Items_Count': [len(selected_items)], 
                        'Total_Value': [int(total_value)], 
                        'Total_Weight': [int(total_weight)]})

def knapsack_local_search(data, itm, val, wgh, capacity, max_iterations=1000, num_initial_solutions=100):
    df = data.copy()
    
    def total_value(solution):
        return sum(df[df[itm].isin(solution)][val])

    def total_weight(solution):
        return sum(df[df[itm].isin(solution)][wgh])

    def generate_random_solution():
        solution, current_weight = [], 0
        items = list(df[itm])
        random.shuffle(items)
        for item in items:
            if current_weight + df[df[itm] == item][wgh].values[0] <= capacity:
                solution.append(item)
                current_weight += df[df[itm] == item][wgh].values[0]
        return solution

    best_solution = max([generate_random_solution() for _ in range(num_initial_solutions)], key=total_value)
    best_value = total_value(best_solution)

    for _ in range(max_iterations):
        neighbor = best_solution.copy()
        if random.random() < 0.5 and neighbor:
            neighbor.remove(random.choice(neighbor))
        else:
            available_items = list(set(df[itm]) - set(neighbor))
            if available_items:
                neighbor.append(random.choice(available_items))
        
        if total_weight(neighbor) <= capacity:
            neighbor_value = total_value(neighbor)
            if neighbor_value > best_value:
                best_solution, best_value = neighbor, neighbor_value

    return pd.DataFrame({
        'Technique' : 'Local Search',
        'Items_Name': [best_solution],
        'Items_Count': [len(best_solution)], 
        'Total_Value': [best_value],
        'Total_Weight': [total_weight(best_solution)]
    })


def knapsack_simulated_annealing(data, itm, val, wgh, capacity, initial_temp=1000, cooling_rate=0.995, 
                                 max_iterations=1000, num_initial_solutions=100):
    df = data.copy()
    
    def total_value(solution):
        return sum(df[df[itm].isin(solution)][val])

    def total_weight(solution):
        return sum(df[df[itm].isin(solution)][wgh])

    def generate_neighbor(solution):
        neighbor = solution.copy()
        if random.random() < 0.5 and len(neighbor) > 0:
            neighbor.remove(random.choice(neighbor))
        else:
            available_items = list(set(df[itm]) - set(neighbor))
            if available_items:
                neighbor.append(random.choice(available_items))
        return neighbor

    def generate_initial_solution():
        solution = []
        for item in df[itm]:
            if random.random() < 0.5 and total_weight(solution + [item]) <= capacity:
                solution.append(item)
        return solution

    best_solution = max([generate_initial_solution() for _ in range(num_initial_solutions)], key=total_value)
    best_value = total_value(best_solution)
    current_solution, current_temp = best_solution.copy(), initial_temp

    for _ in range(max_iterations):
        neighbor = generate_neighbor(current_solution)
        
        if total_weight(neighbor) <= capacity:
            current_value = total_value(current_solution)
            neighbor_value = total_value(neighbor)
            
            if neighbor_value > current_value or random.random() < math.exp((neighbor_value - current_value) / current_temp):
                current_solution = neighbor
                
                if neighbor_value > best_value:
                    best_solution, best_value = neighbor, neighbor_value

        current_temp *= cooling_rate

    
    return pd.DataFrame({
        'Technique' : 'Simulated Annealing',
        'Items_Name': [best_solution],
        'Items_Count': [len(best_solution)], 
        'Total_Value': [best_value],
        'Total_Weight': [total_weight(best_solution)]
    })
    

def knapsack_genetic_algorithm(data, itm, val, wgh, capacity, population_size=100, generations=1000, mutation_rate=0.01):
    df = data.copy()
    
    def create_feasible_individual():
        individual = [0] * len(df)
        total_weight = 0
        while True:
            available_items = [i for i in range(len(df)) if individual[i] == 0 and total_weight + df[wgh][i] <= capacity]
            if not available_items:
                break
            item = random.choice(available_items)
            individual[item] = 1
            total_weight += df[wgh][item]
        return individual

    def fitness(individual):
        return sum(df[val][i] * individual[i] for i in range(len(individual)))

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(individual):
        return [1 - gene if random.random() < mutation_rate else gene for gene in individual]

    def repair(individual):
        total_weight = sum(df[wgh][i] * individual[i] for i in range(len(individual)))
        while total_weight > capacity:
            removable_items = [i for i in range(len(individual)) if individual[i] == 1]
            if not removable_items:
                break
            item_to_remove = random.choice(removable_items)
            individual[item_to_remove] = 0
            total_weight -= df[wgh][item_to_remove]
        return individual

    population = [create_feasible_individual() for _ in range(population_size)]

    for _ in range(generations):
        fitness_scores = [fitness(ind) for ind in population]
        if sum(fitness_scores) > 0:
            parents = random.choices(population, weights=fitness_scores, k=population_size)
        else:
            parents = random.choices(population, k=population_size)  # If all fitness scores are zero, choose randomly
        population = [repair(mutate(crossover(parents[i], parents[i+1]))) for i in range(0, population_size, 2)]

    best_individual = max(population, key=fitness)
    best_value = fitness(best_individual)
    selected_items = [df[itm][i] for i in range(len(best_individual)) if best_individual[i] == 1]
    total_weight = sum(df[wgh][i] for i in range(len(best_individual)) if best_individual[i] == 1)

    return pd.DataFrame({
        'Technique' : 'Genetic Algorithm',
        'Items_Name': [selected_items],
        'Items_Count': [len(selected_items)],
        'Total_Value': [best_value],
        'Total_Weight': [total_weight]
    })


def regression_value(data, val, pop):
    df = data.copy()
    X = df[[pop]]
    y = df[val]
    model = LinearRegression()
    model.fit(X, y)
    
    # Add best fit line as integer
    df['best_fit'] = model.predict(X).round().astype(int)
    df[val] = np.where(df['best_fit']> df[val], df['best_fit'], df[val])
    del df['best_fit']
    return df        

def heuristic_budget_cut(data, itm, val, pop, wgh, capacity_max, capacity_min, cut_step):

    df = regression_value(data, val, pop)
    
    def select_solution(budget, technique):
        if budget == capacity_max:
            s_gr = knapsack_greedy_ratio(df, itm, val, wgh, budget)
            s_ls = knapsack_local_search(df, itm, val, wgh, budget)
            s_sa = knapsack_simulated_annealing(df, itm, val, wgh, budget)
            s_ga = knapsack_genetic_algorithm(df, itm, val, wgh, budget)
            sol = pd.concat([s_gr, s_ls, s_sa, s_ga], ignore_index=True)
        else:
            if technique == 'Greedy Ratio':
                sol = knapsack_greedy_ratio(df, itm, val, wgh, budget)
            elif technique == 'Local Search':
                sol = knapsack_local_search(df, itm, val, wgh, budget)
            elif technique == 'Simulated Annealing':
                sol = knapsack_simulated_annealing(df, itm, val, wgh, budget)
            elif technique == 'Genetic Algorithm':
                sol = knapsack_genetic_algorithm(df, itm, val, wgh, budget)
            else:
                sol = knapsack_greedy_ratio(df, itm, val, wgh, budget)
        sol['Budget'] = budget
        sol = sol.sort_values(by=['Total_Value', 'Items_Count'], ascending=[False, False])
        return sol[['Budget', 'Technique', 'Items_Name', 'Items_Count', 'Total_Value', 'Total_Weight']].head(1), sol['Technique'].iloc[0]

    
    technique = 'Local Search'
    solution_list = []
    for n in range(int((capacity_max-capacity_min)/cut_step)+1):
        budget = int(capacity_max - (n*cut_step))
        # print(budget)
        cut_sol, technique = select_solution(budget, technique)
        solution_list.append(cut_sol)           
    
    solution = pd.concat(solution_list, ignore_index=True)
    solution['Difference'] = solution['Total_Value'].diff()
    return solution[['Budget', 'Items_Name', 'Items_Count', 'Total_Value', 'Difference', 'Total_Weight']]
    