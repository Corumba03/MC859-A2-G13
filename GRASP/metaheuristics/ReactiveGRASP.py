import AbstractGRASP as GRASP
import random
from problems import Evaluator
import Solution

class ReactiveGRASP(GRASP):
    '''Implements the Reactive GRASP metaheuristic.'''

    def __init__(self, obj_function: Evaluator, alpha_pool: list[float], iterations: int = 1, update_freq: int = 10):
        super().__init__(obj_function, self.alpha, iterations)
        
        if alpha_pool is None:
            self.alpha_pool = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            self.alpha_pool = alpha_pool
        self.probabilities = [1/len(alpha_pool)] * len(alpha_pool)
        self.alpha_performance = [0.0] * len(alpha_pool)

        # Select initial alpha randomly
        self.alpha = alpha_pool[0]
        
        # Track cumulative performance of each alpha
        self.alpha_performance = [0.0] * len(alpha_pool)
        self.alpha_counts = [0] * len(alpha_pool)

        self.iterations = iterations
        self.iteration_count = 0

        self.update_freq = update_freq
    
    def select_alpha(self):
        """
        Select an alpha from the pool according to the current probabilities.
        """
        self.alpha = random.choices(self.alpha_pool, weights=self.probabilities)[0]
        return self.alpha
    
    def create_empty_sol(self):
        """Creates and returns an empty solution."""
        return Solution()
    
    def make_CL(self) -> set[int]:
        """Create the initial candidate list (all elements)."""
        return set(range(self.obj_function.get_domain_size()))
    
    def make_RCL(self) -> set[int]:
        """
        Return a subset of CL based on cost ranking and self.alpha.
        Lower cost = better for minimization.
        """
        if not self.CL:
            self.RCL = set()
            return self.RCL

        # Compute cost of adding each element separately in CL to the current solution
        costs = {elem: self.obj_function.evaluate_insertion_cost(elem, self.sol) for elem in self.CL}
        min_cost = min(costs.values())
        max_cost = max(costs.values())
        threshold = min_cost + self.alpha * (max_cost - min_cost)
        
        # Build RCL based on threshold
        self.RCL = {elem for elem, cost in costs.items() if cost <= threshold}
        return self.RCL
    
    def update_CL(self): # TODO: Test with self.CL.discard(chosen) instead since CL will never shrink here for the SCQBF problem
        """
        Remove infeasible candidates (those that violate constraints if added)
        """
        self.CL = {elem for elem in self.CL if self.obj_function.is_feasible(self.sol.insert(elem))}

    def local_search(self, sol: Solution) -> Solution:
        """
        Applies first-improvement local search on the given solution.
        """
        improved = True
        best_sol = sol.copy()
        self.obj_function.evaluate(best_sol)

        while improved:
            improved = False
            # Explore all neighbors (insertion, removal, exchange)
            
            for elem_out in best_sol:
                # Try removal
                neighbor = best_sol.remove(elem_out)
                self.obj_function.evaluate(neighbor)
                if neighbor.cost < best_sol.cost:
                    best_sol = neighbor
                    improved = True
                    break  # first-improvement → restart search
            
            if not improved:
                # Try exchanges
                for elem_out in best_sol:
                    for elem_in in range(self.obj_function.get_domain_size()):
                        if elem_in not in best_sol:
                            neighbor = best_sol.exchange(elem_in, elem_out)
                            self.obj_function.evaluate(neighbor)
                            if neighbor.cost < best_sol.cost:
                                best_sol = neighbor
                                improved = True
                                break
                    if improved:
                        break

            if improved:
                continue

            # Try pure insertions if no improvement yet
            for elem_in in range(self.obj_function.get_domain_size()):
                if elem_in not in best_sol:
                    neighbor = best_sol.insert(elem_in)
                    self.obj_function.evaluate(neighbor)

                    if neighbor.cost < best_sol.cost:
                        best_sol = neighbor
                        improved = True
                        break

        return best_sol
    
    def solve(self) -> Solution:
        """
        Executes Reactive GRASP and returns the best feasible solution found.
        """
        self.best_sol = self.create_empty_sol()
        self.obj_function.evaluate(self.best_sol)

        for i in range(self.iterations):
            self.iteration_count += 1
            self.select_alpha()
            self.sol = self.constructive_heuristic()
            self.obj_function.evaluate(self.sol)
            self.sol = self.local_search(self.sol)

            # Update best solution found
            if self.sol.cost < self.best_sol.cost:
                self.best_sol = self.sol.copy()

            # Update performance of the selected alpha
            alpha_index = self.alpha_pool.index(self.alpha)
            self.alpha_performance[alpha_index] += 1 / (1 + self.sol.cost)
            self.alpha_counts[alpha_index] += 1

            # Periodically update probabilities based on performance
            if (i + 1) % self.update_freq == 0:
                total_performance = sum(self.alpha_performance)
                if total_performance > 0:
                    self.probabilities = [perf / total_performance for perf in self.alpha_performance]
                else:
                    # If no performance recorded, reset to uniform probabilities
                    self.probabilities = [1 / len(self.alpha_pool)] * len(self.alpha_pool)

                # Reset performance metrics for next period
                self.alpha_performance = [0.0] * len(self.alpha_pool)
                self.alpha_counts = [0] * len(self.alpha_pool)

        return self.best_sol  