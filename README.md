# Strategic-Branch-Marketing-Allocation
Des: Implementing the Knapsack Problem algorithm to optimize budget allocation and determine a strategic budget reduction strategy that minimizes impact on returns.

# Instruction
- use python 3.11 for this project
- Install requirements: `pip install -r requirements.txt`
- Read data from google drive, connection has been established, no password required
- Knapsack source code is in ./src/knapsack.py
- To run streamlit app: `streamlit run .src/app.py`

# Pain point

The company faces a dilemma in balancing marketing budget cuts with branch satisfaction and sales targets. This creates tension among cost reduction, effective marketing, branch manager motivation, and meeting sales goals. The current approach of allocating the same marketing budget to all branches is inefficient and fails to address their specific needs and sales potential.

# Objective

Find the biggest possible budget cut with minimal estimated sales impact.

# Approach

-   Perform regression between sales amount and population. If sales amount of a branch is below the best fitted line then value of the branch is the value of best fitted line else sales amount will be the branch value.
-   Knapsack optimization is a problem-solving technique used in resource allocation and decision-making. It involves selecting a combination of items with different values and weights to maximize the total value while staying within a weight limit. This concept is like packing a knapsack with the most valuable items without exceeding its capacity. In budget allocation, this approach applies where branches are items, sales amounts are values, requested budgets are weights, and total budget is threshold.
-   The specific knapsack type used in this project is 0-1. The following is the algorithm used.
    -   Greedy
    -   Local Hill Climbing
    -   Simulated Anneahling
    -   Genetic Algorithm
-    Iterate the process multiple time with reduced threshold.

# Imporovement

- Perform fractural knapsack
- Add Neural Network algorithm
