from random import choices, Random
from GRASP.metaheuristics.AbstractGRASP import AbstractGRASP as GRASP
from GRASP.Solution import Solution
from GRASP.problems import Evaluator

class StandardGRASP(GRASP):
    '''Implements the Standard GRASP metaheuristic.'''
    
    def __init__(
        self, obj_function: Evaluator,  
        alpha_pool: list[float] = None,  
        rand: random.Random = random,
        iterations: int = 1,
        update_freq: int = 10, 
        maximize: bool = True):
        super().__init__(obj_function, alpha=0, iterations=iterations, maximize=maximize)

        if alpha_pool is None:
            self.alpha_pool = [1.0]
        else:
            self.alpha_pool = alpha_pool
            
        self.probabilities = rand.randrange(len(self.alpha_pool))
        self.alpha_performance = [0.0] * len(self.alpha_pool)
            
        # Select initial alpha randomly
        self.alpha = self.alpha_pool[0]
        
        # Track cumulative performance of each alpha
        self.alpha_performance = [0.0] * len(self.alpha)
        self.alpha_counts = [0] * len(self.alpha)

        self.iterations = iterations
        self.iteration_count = 0

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
                if self.is_improvement(neighbor.cost, best_sol.cost):
                    best_sol = neighbor
                    improved = True
                    break
       
            if not improved:
                # Try exchanges
                for elem_out in best_sol:
                    for elem_in in range(self.obj_function.get_domain_size()):
                        if elem_in not in best_sol:
                            neighbor = best_sol.exchange(elem_in, elem_out)
                            self.obj_function.evaluate(neighbor)
                            if self.is_improvement(neighbor.cost, best_sol.cost):
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

                    if self.is_improvement(neighbor.cost, best_sol.cost):
                        best_sol = neighbor
                        improved = True
                        break

        return best_sol
