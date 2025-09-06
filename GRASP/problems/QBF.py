import Evaluator
import Solution

class QBF(Evaluator):
    """
    Quadratic Binary Function evaluator: f(x) = x^T A x
    """

    def __init__(self, n: int, A: list[list[float]]):
        self.size = n
        self.A = A

    def is_feasible(self, sol: Solution) -> bool:
        return True  # All binary vectors are feasible

    # Returns the number of decision variables
    def get_domain_size(self) -> int:
        return self.size

    # Returns the value of the objective function for a given solution
    def evaluate(self, sol: Solution) -> float:
        value = 0.0
        for i in range(self.size):
            row_sum = sum(sol[j] * self.A[i][j] for j in range(self.size))
            value += row_sum * sol[i]
        sol.cost = value
        return value

    # Returns the cost variation of inserting elem into sol
    def evaluate_insertion_cost(self, elem: int, sol: Solution) -> float:
        if sol[elem] == 1:
            return 0.0
        return self.evaluate(sol.insert(elem)) - sol.cost

    # Returns the cost variation of removing elem from sol
    def evaluate_removal_cost(self, elem: int, sol: Solution) -> float:
        if sol[elem] == 0:
            return 0.0
        return self.evaluate(sol.remove(elem)) - sol.cost

    # Returns the cost variation of exchanging elem_out with elem_in in sol
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: Solution) -> float:
        if elem_in == elem_out:
            return 0.0
        if sol[elem_in] == 1:
            return self.evaluate_removal_cost(elem_out, sol)
        if sol[elem_out] == 0:
            return self.evaluate_insertion_cost(elem_in, sol)
        
        new_sol = sol.copy()
        new_sol.exchange(elem_out, elem_in)
        return self.evaluate(sol.exchange(elem_in, elem_out)) - sol.cost

    # ------------------------
    # Utilities
    # ------------------------

    def get_matrix(self) -> list[list[float]]:
        return self.A

    def print_matrix(self):
        for row in self.A:
            print(" ".join(f"{val:.2f}" for val in row))

