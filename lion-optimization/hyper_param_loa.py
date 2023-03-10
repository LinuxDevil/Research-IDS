import numpy as np
from sklearn.model_selection import ParameterGrid

param_grid = {
    "max_iter": [100, 200, 300],
    "num_agents": [10, 20, 30],
    "dimension": [5, 10, 15]
}


def generate_param_combinations(param_grid):
    return list(ParameterGrid(param_grid))


def run_loa(ids_function, params, lb, ub):
    return loa(ids_function, params["max_iter"], params["num_agents"], lb, ub, params["dimension"])


def find_best_hyperparameters(ids_function, param_grid, lb=-10, ub=10):
    param_combinations = generate_param_combinations(param_grid)

    best_params = None
    best_score = np.inf

    for params in param_combinations:
        alpha_pos, alpha_score = run_loa(ids_function, params, lb, ub)

        if alpha_score < best_score:
            best_params = params
            best_score = alpha_score

    return best_params, best_score


def initialize_positions(lower_bound, upper_bound, num_agents, dimension):
    return np.random.uniform(lower_bound, upper_bound, size=(num_agents, dimension))


def update_alpha_beta_delta_positions(positions, fitness, best_alpha_pos, best_alpha_score, best_beta_pos, best_beta_score, best_delta_pos, best_delta_score):
    alpha_mask = fitness < best_alpha_score
    best_alpha_score = np.where(alpha_mask, fitness, best_alpha_score)
    best_alpha_pos = np.where(alpha_mask.reshape(-1, 1), positions, best_alpha_pos)

    beta_mask = (fitness > best_alpha_score) & (fitness < best_beta_score)
    best_beta_score = np.where(beta_mask, fitness, best_beta_score)
    best_beta_pos = np.where(beta_mask.reshape(-1, 1), positions, best_beta_pos)

    delta_mask = (fitness > best_alpha_score) & (fitness > best_beta_score) & (fitness < best_delta_score)
    best_delta_score = np.where(delta_mask, fitness, best_delta_score)
    best_delta_pos = np.where(delta_mask.reshape(-1, 1), positions, best_delta_pos)

    return best_alpha_pos, best_alpha_score, best_beta_pos, best_beta_score, best_delta_pos, best_delta_score


def update_agent_positions(positions, best_alpha_pos, best_beta_pos, best_delta_pos):
    random_1 = np.random.rand(len(positions), len(positions[0]))
    random_2 = np.random.rand(len(positions), len(positions[0]))
    agent_movement_1 = 2 * random_1 - 1
    agent_movement_2 = 2 * random_2
    distance_from_delta = np.abs(agent_movement_2 * best_delta_pos - positions)
    position_change_1 = best_delta_pos - agent_movement_1 * distance_from_delta
    position_change_2 = best_alpha_pos - agent_movement_1 * distance_from_delta
    position_change_3 = (best_alpha_pos + best_beta_pos + best_delta_pos) / 3
    positions = (position_change_1 + position_change_2 + position_change_3) / 3
    return positions


def apply_position_bounds(positions, lower_bound, upper_bound):
    return np.clip(positions, lower_bound, upper_bound)


def loa(ids_function, max_iterations, num_agents, lower_bound, upper_bound, dimension):
    # Initialize variables
    best_alpha_pos = np.zeros(dimension)
    best_alpha_score = np.inf
    best_beta_pos = np.zeros(dimension)
    best_beta_score = np.inf
    best_delta_pos = np.zeros(dimension)
    best_delta_score = np.inf
    positions = initialize_positions(lower_bound, upper_bound, num_agents, dimension)

    # Main loop
    for iteration in range(max_iterations):
        # Calculate fitness for each agent
        fitness = np.apply_along_axis(ids_function, 1, positions)

        # Update alpha, beta, and delta positions
        (best_alpha_pos, best_alpha_score, best_beta_pos, best_beta_score, best_delta_pos, best_delta_score) = update_alpha_beta_delta_positions(
            positions, fitness, best_alpha_pos, best_alpha_score, best_beta_pos, best_beta_score, best_delta_pos, best_delta_score)

        # Update the positions of agents
        positions = update_agent_positions(positions, best_alpha_pos, best_beta_pos, best_delta_pos)

        # Apply bounds to the positions
        positions = apply_position_bounds(positions, lower_bound, upper_bound)

    # Return the best solution
    return best_alpha_pos, best_alpha_score
