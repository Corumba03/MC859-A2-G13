import Evaluator

class QBF(Evaluator):
    """
    Quadratic Binary Function evaluator: f(x) = x^T A x
    """

    def __init__(self, n: int, A: list[list[float]]):
        self.size = n
        self.A = A

    # Returns the number of decision variables
    def get_domain_size(self) -> int:
        return self.size

    # Returns the value of the objective function for a given solution
    def evaluate(self, sol: set) -> float:
        value = 0.0
        for i in range(self.size):
            row_sum = sum(sol[j] * self.A[i][j] for j in range(self.size))
            value += row_sum * sol[i]
        return value

    # Returns the cost variation of inserting elem into sol
    def evaluate_insertion_cost(self, elem: int, sol: set) -> float:
        if sol[elem] == 1:
            return 0.0
        return self.evaluate(sol | {elem}) - self.evaluate(sol)

    # Returns the cost variation of removing elem from sol
    def evaluate_removal_cost(self, elem: int, sol: set) -> float:
        if sol[elem] == 0:
            return 0.0
        return self.evaluate(sol - {elem}) - self.evaluate(sol)

    # Returns the cost variation of exchanging elem_out with elem_in in sol
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: set) -> float:
        if elem_in == elem_out:
            return 0.0
        if sol[elem_in] == 1:
            return self.evaluate_removal_cost(elem_out, sol)
        if sol[elem_out] == 0:
            return self.evaluate_insertion_cost(elem_in, sol)
        
        return self.evaluate((sol - {elem_out}) | {elem_in}) - self.evaluate(sol)

    # ------------------------
    # Utilities
    # ------------------------

    def get_matrix(self) -> list[list[float]]:
        return self.A

    def print_matrix(self):
        for row in self.A:
            print(" ".join(f"{val:.2f}" for val in row))

