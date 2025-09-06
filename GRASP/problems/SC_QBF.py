import Evaluator
import SetCover
import QBF

class SCQBF(Evaluator):
    '''
    This class implements the Set Covering Quadratic Binary Function (SC-QBF) problem.
    The objective is to select a subset of sets that covers all elements while maximizing
    a quadratic binary function defined by a matrix A.
    '''
    def __init__(self, n: int, A: list[list[float]], sets: list[set[int]]):
        self.n = n
        self.A = A
        self.sets = sets

        # Calls the subproblem constructors (Set Cover and QBF)
        self.SC = SetCover.SetCover(sets, n)
        self.QBF = QBF.QBF(n, A)
    
    def is_feasible(self, sol: set) -> bool:
        return self.SC.is_feasible(sol)

    
    # If the solution is feasible, return its QBF value; otherwise, return -inf (infeasible)
    def evaluate(self, sol: set) -> float:
        if self.is_feasible(sol):
            return self.QBF.evaluate(sol)
        return float("inf")

    def evaluate_insertion_cost(self, elem: int, sol: set) -> float:
        if self.is_feasible(sol | {elem}):
            return self.QBF.evaluate_insertion_cost(elem, sol)
        return float("inf")


    def evaluate_removal_cost(self, elem: int, sol: set) -> float:
        if self.is_feasible(sol - {elem}):
            return self.QBF.evaluate_removal_cost(elem, sol)
        return float("inf")

    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, sol: set) -> float:
        if self.is_feasible((sol - {elem_out}) | {elem_in}):
            return self.QBF.evaluate_exchange_cost(elem_in, elem_out, sol)
        return float("inf")

    def get_domain_size(self) -> int:
        return self.n

