from abc import ABC, abstractmethod
from solutions import Solution  # Assuming Solution class is in solution.py

'''
The Evaluator interface gives to a problem the required functionality to
obtain a mapping of a solution (n-dimensional array of elements of generic
type E (domain)) to a Double (image). It is a useful representation of an
objective function for an optimization problem.

This Python port is based on the original Java implementation by 
ccavellucci and fusberti
'''


class Evaluator(ABC):
    """
    Abstract class representing an objective function for an optimization problem.
    Provides methods to evaluate a solution and the effect of inserting, removing,
    or exchanging elements in a solution.
    """

    @abstractmethod
    def getDomainSize(self) -> int:
        """Returns the size of the problem domain (number of decision variables)."""
        pass

    @abstractmethod
    def evaluate(self, sol: Solution) -> float:
        """
        Returns the objective function value of a solution.
        """
        pass

    @abstractmethod
    def evaluateInsertionCost(self, elem, sol: Solution) -> float:
        """
        Returns the cost variation of inserting `elem` into `sol`.
        """
        pass

    @abstractmethod
    def evaluateRemovalCost(self, elem, sol: Solution) -> float:
        """
        Returns the cost variation of removing `elem` from `sol`.
        """
        pass

    @abstractmethod
    def evaluateExchangeCost(self, elem_in, elem_out, sol: Solution) -> float:
        """
        Returns the cost variation of exchanging `elem_out` with `elem_in` in `sol`.
        """
        pass



