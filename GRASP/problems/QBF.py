from solutions import Solution
import Evaluator

class QBF(Evaluator):
    """
    Quadratic Binary Function evaluator: f(x) = x^T A x
    """

    def __init__(self, n: int, A: list[list[float]]):
        self.size = n
        self.A = A
        self.variables = [0.0] * n  # current solution vector

    # ------------------------
    # Evaluator methods
    # ------------------------

    # Returns the number of decision variables
    def getDomainSize(self) -> int:
        return self.size

    # Returns the value of the objective function for a given solution
    def evaluate(self, sol: Solution) -> float:
        self.set_variables(sol)
        sol.cost = self.evaluateQBF()
        return sol.cost

    # Returns the cost variation of inserting elem into sol
    def evaluateInsertionCost(self, elem: int, sol: Solution) -> float:
        self.set_variables(sol)
        return self.evaluateInsertionQBF(elem)

    # Returns the cost variation of removing elem from sol
    def evaluateRemovalCost(self, elem: int, sol: Solution) -> float:
        self.set_variables(sol)
        return self.evaluateRemovalQBF(elem)

    # Returns the cost variation of exchanging elem_out with elem_in in sol
    def evaluateExchangeCost(self, elem_in: int, elem_out: int, sol: Solution) -> float:
        self.set_variables(sol)
        return self.evaluateExchangeQBF(elem_in, elem_out)

    # ------------------------
    # QBF-specific evaluations
    # ------------------------

    # Creates a binary vector representation of the solution
    def set_variables(self, sol: Solution):
        self.reset_variables()
        if sol and len(sol) > 0:
            for elem in sol:
                self.variables[elem] = 1.0

    # Resets the binary vector to all zeros
    def reset_variables(self):
        self.variables = [0] * self.size
    
    # Full evaluation of the QBF f(x) = x^T A x
    def evaluateQBF(self) -> float:
        total = 0.0
        for i in range(self.size):
            row_sum = sum(self.variables[j] * self.A[i][j] for j in range(self.size))
            total += row_sum * self.variables[i]
        return total

    # Incremental evaluations
    def evaluateInsertionQBF(self, i: int) -> float:
        if self.variables[i] == 1:
            return 0.0
        return self.evaluateContributionQBF(i)
    
    def evaluateRemovalQBF(self, i: int) -> float:
        if self.variables[i] == 0:
            return 0.0
        return -self.evaluateContributionQBF(i)
    
    def evaluateContributionQBF(self, i: int) -> float:
        """Incremental contribution of a single variable to the QBF."""
        total = sum(
            self.variables[j] * (self.A[i][j] + self.A[j][i])
            for j in range(self.size) if j != i
        )
        total += self.A[i][i]
        return total

    '''
    Incremental evaluation of exchanging elem_out with elem_in.
    This is done by adding the contribution of elem_in and removing
    the contribution of elem_out, and then adjusting for the interaction
    between the two elements, which is subtracted twice in the previous step.
    '''
    def evaluateExchangeQBF(self, elem_in: int, elem_out: int) -> float:
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1:
            return self.evaluateRemovalQBF(elem_out)
        if self.variables[elem_out] == 0:
            return self.evaluateInsertionQBF(elem_in)

        total = self.evaluateContributionQBF(elem_in)
        total -= self.evaluateContributionQBF(elem_out)
        total -= (self.A[elem_in][elem_out] + self.A[elem_out][elem_in])

        return total

    # ------------------------
    # Utilities
    # ------------------------

    def print_matrix(self):
        for row in self.A:
            print(" ".join(f"{val:.2f}" for val in row))

