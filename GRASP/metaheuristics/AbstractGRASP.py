import random
from abc import ABC, abstractmethod
from ..problems import Evaluator  # use relative import to problems

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
        self.iterations = iterations + 1  # +1 to account for 0-based indexing

        self.best_cost = float("-inf") if maximize else float("inf")
        self.cost = float("-inf") if maximize else float("inf")
        self.last_cost = None

        self.best_sol = None
        self.sol = None

        self.CL = []   # Candidate List
        self.RCL = []  # Restricted Candidate List

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
        # Debug: print initial CL, all sets, and union of all sets
        if hasattr(self.obj_function, 'SC') and hasattr(self.obj_function.SC, 'sets'):
            union_all_sets = set()
            for s in self.obj_function.SC.sets:
                union_all_sets.update(s)

        """Builds a feasible solution using the GRASP constructive heuristic."""
        self.sol = self.create_empty_sol()
        self.sol.cost = self.obj_function.evaluate(self.sol, allow_partial=True)
        self.last_cost = self.sol.cost
        self.CL = self.make_CL()

        while True:
            self.update_CL()
            if not self.CL:
                break

            # Compute insertion costs
            deltas = {c: self.obj_function.evaluate_insertion_cost(c, self.sol) for c in self.CL}
            min_cost = min(deltas.values())
            max_cost = max(deltas.values())

            # Build RCL based on alpha
            if self.maximize:
                threshold = max_cost - self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, delta in deltas.items() if delta >= threshold]
            else:
                threshold = min_cost + self.alpha * (max_cost - min_cost)
                self.RCL = [c for c, delta in deltas.items() if delta <= threshold]
            if not self.RCL:
                break

            in_cand = self.rng.choice(self.RCL)
            self.CL.remove(in_cand)
            self.sol.add(in_cand)

            new_cost = self.obj_function.evaluate(self.sol, allow_partial=True)
            self.last_cost = new_cost
            self.RCL.clear()

            # Stop if cost does not improve AND solution is feasible
            if self.constructive_stop_criteria(new_cost) and self.obj_function.is_feasible(self.sol):
                break

        # Guarantee feasibility: explicitly check coverage and greedily add sets until all elements are covered
        if hasattr(self.obj_function, 'SC') and hasattr(self.obj_function.SC, 'coverage'):
            covered = set(self.obj_function.SC.coverage(self.sol))
            all_elements = set(range(self.obj_function.SC.num_elements))
            # If not feasible, reconstruct CL to include all unused sets
            if covered != all_elements:
                used = set(self.sol.elements)
                all_sets = set(range(len(self.obj_function.SC.sets)))
                self.CL = all_sets - used
            while covered != all_elements and self.CL:
                # Pick a candidate that covers the most uncovered elements
                best_cand = max(self.CL, key=lambda c: len(set(self.obj_function.SC.sets[c]) - covered))
                self.CL.remove(best_cand)
                self.sol.add(best_cand)
                covered.update(self.obj_function.SC.sets[best_cand])
                self.obj_function.evaluate(self.sol, allow_partial=True)

        return self.sol

    def solve(self):
        """Executes GRASP and returns a list of best solutions per iteration."""
        self.best_sol = None
        best_solutions = []

        for i in range(self.iterations):
            self.sol = self.constructive_heuristic()
            # Final evaluation: only feasible solutions count

            self.obj_function.evaluate(self.sol, allow_partial=False)

            last_feasible = self.sol.copy()
            self.sol = self.local_search(self.sol)
            self.obj_function.evaluate(self.sol, allow_partial=False)

            # Safeguard: if local search returns infeasible, revert to last feasible
            if not self.obj_function.is_feasible(self.sol):
                self.sol = last_feasible
                self.obj_function.evaluate(self.sol, allow_partial=False)

            # Only consider feasible, non-empty solutions
            if self.obj_function.is_feasible(self.sol) and len(self.sol.elements) > 0:
                if self.best_sol is None:
                    self.best_sol = self.sol.copy()
                elif (self.maximize and self.sol.cost > self.best_sol.cost):
                    self.best_sol = self.sol.copy()
                elif (not self.maximize and self.sol.cost < self.best_sol.cost):
                    self.best_sol = self.sol.copy()
                    self.best_sol = self.sol.copy()

                best_solutions.append(self.sol.copy())

        return best_solutions

    def constructive_stop_criteria(self, new_cost):
        """Stops when adding new candidates no longer improves the solution."""
        if self.last_cost is None:
            return False
        if self.maximize:
            return new_cost <= self.last_cost
        else:
            return new_cost >= self.last_cost
