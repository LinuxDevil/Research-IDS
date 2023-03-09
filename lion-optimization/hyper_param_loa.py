import numpy as np
from sklearn.model_selection import ParameterGrid

param_grid = {
    "max_iter": [100, 200, 300],
    "num_agents": [10, 20, 30],
    "dimension": [5, 10, 15]
}

def optimize_loa(ids_function, param_grid):
    """
    Optimizes the LOA algorithm for an IDS function by tuning the hyperparameters.

    Parameters:
    ids_function (function): The IDS function to optimize.
    param_grid (dict): The grid of hyperparameters to search.

    Returns:
    tuple: A tuple containing the best hyperparameters found by the optimization and their corresponding score.
    """
    # Define the bounds of the search space
    lb = -10
    ub = 10

    # Generate all possible combinations of hyperparameters
    param_combinations = list(ParameterGrid(param_grid))

    # Initialize variables
    best_params = None
    best_score = np.inf

    # Loop over all hyperparameter combinations
    for params in param_combinations:
        # Run the LOA algorithm with the current hyperparameters
        alpha_pos, alpha_score = loa(ids_function, params["max_iter"], params["num_agents"], lb, ub,
                                     params["dimension"])

        # Update the best hyperparameters if a better solution was found
        if alpha_score < best_score:
            best_params = params
            best_score = alpha_score

    # Return the best hyperparameters found
    return best_params, best_score


def loa(ids_function, max_iterations, num_agents, lower_bound, upper_bound, dimension):
    """
    Runs the Lion Optimization Algorithm (LOA) to optimize an Intrusion Detection System (IDS) function.

    Parameters:
    ids_function (function): The IDS function to optimize.
    max_iterations (int): The maximum number of iterations to run the LOA algorithm.
    num_agents (int): The number of agents to use in the LOA algorithm.
    lower_bound (float): The lower bound for the search space.
    upper_bound (float): The upper bound for the search space.
    dimension (int): The dimensionality of the search space.

    Returns:
    tuple: A tuple containing the best position found by the LOA algorithm and its corresponding score.
    """
    # Initialize variables
    best_alpha_pos = np.zeros(dimension)
    best_alpha_score = np.inf
    best_beta_pos = np.zeros(dimension)
    best_beta_score = np.inf
    best_delta_pos = np.zeros(dimension)
    best_delta_score = np.inf
    positions = np.random.uniform(lower_bound, upper_bound, size=(num_agents, dimension))

    # Main loop
    for iteration in range(max_iterations):
        # Calculate fitness for each agent
        fitness = np.apply_along_axis(ids_function, 1, positions)

        # Update alpha, beta, and delta positions
        alpha_mask = fitness < best_alpha_score
        best_alpha_score = np.where(alpha_mask, fitness, best_alpha_score)
        best_alpha_pos = np.where(alpha_mask.reshape(-1, 1), positions, best_alpha_pos)

        beta_mask = (fitness > best_alpha_score) & (fitness < best_beta_score)
        best_beta_score = np.where(beta_mask, fitness, best_beta_score)
        best_beta_pos = np.where(beta_mask.reshape(-1, 1), positions, best_beta_pos)

        delta_mask = (fitness > best_alpha_score) & (fitness > best_beta_score) & (fitness < best_delta_score)
        best_delta_score = np.where(delta_mask, fitness, best_delta_score)
        best_delta_pos = np.where(delta_mask.reshape(-1, 1), positions, best_delta_pos)

        # Update the positions of agents
        random_1 = np.random.rand(num_agents, dimension)
        random_2 = np.random.rand(num_agents, dimension)
        agent_movement_1 = 2 * random_1 - 1
        agent_movement_2 = 2 * random_2
        distance_from_delta = np.abs(agent_movement_2 * best_delta_pos - positions)
        position_change_1 = best_delta_pos - agent_movement_1 * distance_from_delta
        position_change_2 = best_alpha_pos - agent_movement_1 * distance_from_delta
        position_change_3 = (best_alpha_pos + best_beta_pos + best_delta_pos) / 3
        positions = (position_change_1 + position_change_2 + position_change_3) / 3

        # Apply bounds to the positions
        positions = np.clip(positions, lower_bound, upper_bound)

    # Return the best solution
    return best_alpha_pos, best_alpha_score
