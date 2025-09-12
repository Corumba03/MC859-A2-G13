from random import choices, Random
from GRASP.metaheuristics.AbstractGRASP import AbstractGRASP as GRASP
from GRASP.Solution import Solution
from GRASP.problems import Evaluator

class ReactiveGRASP(GRASP):
    '''Implements the Reactive GRASP metaheuristic.'''

    def __init__(
            self, obj_function: Evaluator, 
            alpha_pool: list[float] = None, 
            iterations: int = 1, 
            update_freq: int = 10, 
            maximize: bool = True,
            search_type: str = 'first',
            constructive_type: str = 'std'):
        super().__init__(obj_function, alpha=0, iterations=iterations, maximize=maximize, constructive_type=constructive_type)
        
        if alpha_pool is None:
            self.alpha_pool = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            self.alpha_pool = alpha_pool
        self.probabilities = [1/len(self.alpha_pool)] * len(self.alpha_pool)
        self.alpha_performance = [0.0] * len(self.alpha_pool)

        # Select initial alpha randomly
        self.alpha = self.alpha_pool[0]
        
        # Track cumulative performance of each alpha
        self.alpha_performance = [0.0] * len(self.alpha_pool)
        self.alpha_counts = [0] * len(self.alpha_pool)

        self.iterations = iterations
        self.iteration_count = 0
        self.search_type = search_type  # 'first' or 'best'
        self.constructive_type = constructive_type  # 'std', 'max_coverage', 'cost_ratio'

        self.update_freq = update_freq

    
    def select_alpha(self):
        """
        Select an alpha from the pool according to the current probabilities.
        """
        self.alpha = choices(self.alpha_pool, weights=self.probabilities)[0]
        return self.alpha
    
    def create_empty_sol(self):
        """Creates and returns an empty solution."""
        return Solution(maximize=self.maximize)

    
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
        deltas = {elem: self.obj_function.evaluate_insertion_cost(elem, self.sol) for elem in self.CL}
        min_cost = min(deltas.values())
        max_cost = max(deltas.values())
        
        # Build RCL based on threshold
        if self.maximize:
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta >= threshold}
        else:
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta <= threshold}

        return self.RCL
    
    def update_CL(self): # TODO: Test with self.CL.discard(chosen) instead since CL will never shrink here for the SCQBF problem
        """
        Remove infeasible candidates (those that violate constraints if added)
        """
        # All candidates are feasible
        pass 

    def is_improvement(self, new_cost: float, current_cost: float) -> bool:
        """
        Determines if the new cost is an improvement over the current cost.
        """
        if self.maximize:
            return new_cost > current_cost
        else:
            return new_cost < current_cost
    
    def solve(self):
        """
        Executes Reactive GRASP and returns a list of best solutions per iteration.
        Ensures correct maximization/minimization logic for best solution selection.
        """
        self.best_sol = self.create_empty_sol()
        self.obj_function.evaluate(self.best_sol)
        best_solutions = []


        for i in range(self.iterations):
            self.iteration_count += 1
            self.select_alpha()
            if self.constructive_type == 'std':
                self.sol = self.constructive_heuristic()
            elif self.constructive_type == 'max_coverage':
                self.sol = self.constructive_greedy_max_coverage()
            elif self.constructive_type == 'cost_ratio':
                self.sol = self.constructive_greedy_cost_ratio()
            self.obj_function.evaluate(self.sol)

            # Local search
            if self.search_type == 'first':
                local_sol = self.local_search_first(self.sol)
            else:
                local_sol = self.local_search_best(self.sol)
            self.obj_function.evaluate(local_sol)

            # Choose the best between constructive and local search for this iteration
            candidates = []
            for candidate in [self.sol, local_sol]:
                feasible = self.obj_function.is_feasible(candidate)
                self.obj_function.evaluate(candidate)
                if feasible and candidate.elements:
                    candidates.append(candidate)

            if candidates:
                # Pick the best candidate for this iteration
                if self.maximize:
                    iter_best = max(candidates, key=lambda s: s.cost)
                else:
                    iter_best = min(candidates, key=lambda s: s.cost)

                # Update global best if needed
                if (len(self.best_sol.elements) == 0 or
                    (self.maximize and iter_best.cost > self.best_sol.cost) or
                    (not self.maximize and iter_best.cost < self.best_sol.cost)):
                    self.best_sol = iter_best.copy()

                best_solutions.append(self.best_sol.copy())
            else:
                # No feasible solution found this iteration, append current best
                best_solutions.append(self.best_sol.copy())

            # Update alpha performance
            alpha_index = self.alpha_pool.index(self.alpha)
            # Use the cost of the best solution found this iteration for performance
            perf_cost = iter_best.cost if candidates else self.best_sol.cost
            self.alpha_performance[alpha_index] += 1 / (1 + abs(perf_cost))
            self.alpha_counts[alpha_index] += 1

            # Periodically update probabilities based on performance
            if (i + 1) % self.update_freq == 0:
                total_performance = sum(self.alpha_performance)
                if total_performance > 0:
                    self.probabilities = [perf / total_performance for perf in self.alpha_performance]
                else:
                    self.probabilities = [1 / len(self.alpha_pool)] * len(self.alpha_pool)
                self.alpha_performance = [0.0] * len(self.alpha_pool)
                self.alpha_counts = [0] * len(self.alpha_pool)

        return best_solutions