import numpy as np


class GreyWolfOptimizer:
    def __init__(self, obj_function, search_space, pack_size, n_iter):
        self.obj_function = obj_function
        self.search_space = search_space
        self.pack_size = pack_size
        self.n_iter = n_iter
        self.dim = len(search_space)

    def optimize(self):
        # Initialize pack positions
        pack_positions = np.array(
            [np.random.uniform(space[0], space[1], self.pack_size) for space in self.search_space]).T

        # Calculate objective function values for each position
        obj_values = np.array([self.obj_function(pos) for pos in pack_positions])

        # Find the initial alpha, beta, and delta wolves
        alpha_idx, beta_idx, delta_idx = np.argsort(obj_values)[:3]

        alpha = pack_positions[alpha_idx]
        beta = pack_positions[beta_idx]
        delta = pack_positions[delta_idx]

        # Start GWO iterations
        for iter_idx in range(self.n_iter):
            a = 2 - iter_idx * (2 / self.n_iter)  # Linearly decreasing 'a' parameter

            for i in range(self.pack_size):
                # Update positions of the pack members
                for j in range(self.dim):
                    r1, r2 = np.random.rand(2)  # Random coefficients

                    # Calculate the updating components for each of the three leading wolves
                    A_alpha = 2 * a * r1 - a
                    A_beta = 2 * a * r1 - a
                    A_delta = 2 * a * r1 - a

                    C_alpha = 2 * r2
                    C_beta = 2 * r2
                    C_delta = 2 * r2

                    D_alpha = abs(C_alpha * alpha[j] - pack_positions[i, j])
                    D_beta = abs(C_beta * beta[j] - pack_positions[i, j])
                    D_delta = abs(C_delta * delta[j] - pack_positions[i, j])

                    # Update the position of the current pack member
                    pack_positions[i, j] = (alpha[j] - A_alpha * D_alpha + beta[j] - A_beta * D_beta +
                                            delta[j] - A_delta * D_delta) / 3

                    # Clip the updated position to the search space bounds
                    pack_positions[i, j] = np.clip(pack_positions[i, j], self.search_space[j][0],
                                                   self.search_space[j][1])
                # Calculate the updated objective function value
                updated_obj_value = self.obj_function(pack_positions[i])

                # Update the alpha, beta, and delta wolves if necessary
                if updated_obj_value < obj_values[alpha_idx]:
                    delta_idx = beta_idx
                    beta_idx = alpha_idx
                    alpha_idx = i
                elif updated_obj_value < obj_values[beta_idx]:
                    delta_idx = beta_idx
                    beta_idx = i
                elif updated_obj_value < obj_values[delta_idx]:
                    delta_idx = i

                # Update the objective function value for the current pack member
                obj_values[i] = updated_obj_value

                # Update the positions of the alpha, beta, and delta wolves
                alpha = pack_positions[alpha_idx]
                beta = pack_positions[beta_idx]
                delta = pack_positions[delta_idx]

        # Return the best solution found
        return alpha
