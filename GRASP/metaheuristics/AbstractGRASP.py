import random
from problems import Evaluator
from abc import ABC, abstractmethod

'''
  Abstract class for metaheuristic GRASP (Greedy Randomized Adaptive Search
  Procedure). It consider a minimization problem.
  
  
  This Python port is based on the original Java implementation by
  ccavellucci, fusberti

'''
class AbstractGRASP(ABC):
    """
    Abstract class for metaheuristic GRASP (Greedy Randomized Adaptive Search Procedure).
    This version considers a minimization problem.
    """

    verbose = True
    rng = random.Random(0)

    def __init__(self, obj_function: Evaluator, alpha: float = 0.0, iterations: int = 1, maximize: bool = True):
        self.obj_function = obj_function
        self.alpha = alpha
        self.iterations = iterations

        self.best_cost = float("-inf") if maximize else float("inf")
        self.cost = float("-inf") if maximize else float("inf")


        self.best_sol = None
        self.sol = None

        self.CL = set()
        self.RCL = set()

        self.maximize = maximize

    # --- Abstract methods (must be implemented in subclasses) ---
    @abstractmethod
    def make_CL(self):
        """Creates the Candidate List of elements to enter the solution."""
        pass

    @abstractmethod
    def make_RCL(self):
        """Creates the Restricted Candidate List of elements to enter the solution."""
        pass

    @abstractmethod
    def update_CL(self):
        """Updates the Candidate List according to the current solution."""
        pass

    @abstractmethod
    def create_empty_sol(self):
        """Creates and returns an empty solution."""
        pass

    @abstractmethod
    def local_search(self):
        """Performs local search and returns a locally optimal solution."""
        pass

    # --- Concrete methods ---
    def constructive_heuristic(self):
        """Builds a feasible solution using the GRASP constructive heuristic."""
        self.CL = self.make_CL()
        self.RCL = self.make_RCL()
        self.sol = self.create_empty_sol()
        self.cost = float("inf")

        while not self.constructive_stop_criteria():
            max_cost = float("-inf")
            min_cost = float("inf")

            self.cost = self.obj_function.evaluate(self.sol)
            self.update_CL()

            if not self.CL:
                break

            # Evaluate candidate insertions
            deltas = {c: self.obj_function.evaluate_insertion_cost(c, self.sol) for c in self.CL}
            min_cost = min(deltas.values())
            max_cost = max(deltas.values())

            # Build RCL with candidates within threshold
            if self.maximize:
                threshold = max_cost - self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, delta in deltas.items() if delta >= threshold]
            else:
                threshold = min_cost + self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, delta in deltas.items() if delta <= threshold]


            # Pick a random candidate from RCL
            if not self.RCL: 
                """This ensures a strictly greedy choice if RCL is empty. 
                Can happen with a very low alpha. Not good for diversification."""
                break

            in_cand = self.rng.choice(self.RCL)
            self.CL.remove(in_cand)
            self.sol.add(in_cand)
            self.obj_function.evaluate(self.sol)
            self.RCL.clear()

        return self.sol

    def solve(self):
        """Executes GRASP and returns the best feasible solution found."""
        self.best_sol = self.create_empty_sol()

        for i in range(self.iterations):
            self.sol = self.constructive_heuristic()
            self.local_search()

            if (self.maximize and self.sol.cost > self.best_sol.cost) or \
                (not self.maximize and self.sol.cost < self.best_sol.cost):

                self.best_sol = self.sol.copy()
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}, cost = {self.best_sol.cost}")


        return self.best_sol

    def constructive_stop_criteria(self):
        """Stops when adding new candidates no longer improves the solution."""
        if self.maximize:
            return self.cost >= self.sol.cost
        else:
            return self.cost <= self.sol.cost
