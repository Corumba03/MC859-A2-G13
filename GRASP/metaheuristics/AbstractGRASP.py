import random
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

    def __init__(self, obj_function, alpha: float = 0.0, iterations: int = 1):
        self.obj_function = obj_function
        self.alpha = alpha
        self.iterations = iterations

        self.best_cost = float("inf")
        self.cost = float("inf")

        self.best_sol = None
        self.sol = None

        self.CL = []
        self.RCL = []

    # --- Abstract methods (must be implemented in subclasses) ---
    @abstractmethod
    def makeCL(self):
        """Creates the Candidate List of elements to enter the solution."""
        pass

    @abstractmethod
    def makeRCL(self):
        """Creates the Restricted Candidate List of elements to enter the solution."""
        pass

    @abstractmethod
    def updateCL(self):
        """Updates the Candidate List according to the current solution."""
        pass

    @abstractmethod
    def createEmptySol(self):
        """Creates and returns an empty solution."""
        pass

    @abstractmethod
    def localSearch(self):
        """Performs local search and returns a locally optimal solution."""
        pass

    # --- Concrete methods ---
    def constructiveHeuristic(self):
        """Builds a feasible solution using the GRASP constructive heuristic."""
        self.CL = self.makeCL()
        self.RCL = self.makeRCL()
        self.sol = self.createEmptySol()
        self.cost = float("inf")

        while not self.constructiveStopCriteria():
            max_cost = float("-inf")
            min_cost = float("inf")

            self.cost = self.obj_function.evaluate(self.sol)
            self.updateCL()

            # Evaluate candidate insertions
            for c in self.CL:
                delta = self.obj_function.evaluateInsertionCost(c, self.sol)
                min_cost = min(min_cost, delta)
                max_cost = max(max_cost, delta)

            # Build RCL with candidates within threshold
            self.RCL = [
                c for c in self.CL
                if self.obj_function.evaluateInsertionCost(c, self.sol)
                <= min_cost + self.alpha * (max_cost - min_cost)
            ]

            # Pick a random candidate from RCL
            in_cand = self.rng.choice(self.RCL)
            self.CL.remove(in_cand)
            self.sol.add(in_cand)
            self.obj_function.evaluate(self.sol)
            self.RCL.clear()

        return self.sol

    def solve(self):
        """Executes GRASP and returns the best feasible solution found."""
        self.best_sol = self.createEmptySol()

        for i in range(self.iterations):
            self.constructiveHeuristic()
            self.localSearch()

            if self.best_sol.cost > self.sol.cost:
                self.best_sol = self.sol.copy()
                if self.verbose:
                    print(f"(Iter. {i}) BestSol = {self.best_sol}")

        return self.best_sol

    def constructiveStopCriteria(self):
        """Stops when adding new candidates no longer improves the solution."""
        return not (self.cost > self.sol.cost)
