__package__ = "problems"

from abc import ABC, abstractmethod
import solutions.Solution as Solution

'''
The Evaluator interface gives to a problem the required functionality to
obtain a mapping of a solution (n-dimensional array of elements of generic
type E (domain)) to a Double (image). It is a useful representation of an
objective function for an optimization problem.

This Python port is based on the original Java implementation by 
ccavellucci and fusberti
'''
class Evaluator(ABC):
    '''
    Gives the size of the problem domain. Typically this is the number of
	decision variables of an optimization problem.
	
    Returns:
	    the size of the problem domain.
    '''
    @abstractmethod
    def getDomainSize(self):
        pass

    '''
    The evaluating function is responsible for returning the mapping value of
	a solution.
	 
    Attributes:
	    sol
	      the solution under evaluation.

    Returns:
	    the evaluation of a solution.
     '''
    
    @abstractmethod
    def evaluate(self, sol):
        pass

    '''
    Evaluates the cost variation of inserting an element into a solution
	according to an objective function.
	
    Attributes:
	    elem
	        the element under consideration for insertion.
	    sol
	        the solution for which the element insertion is being
            evaluated.

    Returns:
	    the cost variation resulting from the element insertion into the
	    solution.
    '''

    @abstractmethod
    def evaluateInsertionCost(self, elem, sol):
        pass

    '''
    Evaluates the cost variation of removing an element into a solution
	according to an objective function.
    
    Attributes:
        elem
            the element under consideration for removal.
        sol
            the solution for which the element removal is being evaluated.

    Returns:
        the cost variation resulting from the element removal from the solution.
    '''
    @abstractmethod
    def evaluateRemovalCost(self, elem, sol):
        pass

    '''
    Evaluates the cost variation of exchanging candidates, one being
	considered to enter the solution (elemIn) and the other being considered
	for removal (elemOut).

    Attributes:
        elemIn
            the element under consideration for insertion.
        elemOut
            the element under consideration for removal.
        sol
            the solution for which the element exchange is being evaluated.
    
    Returns:
        the cost variation resulting from the elements exchange.
    '''
    @abstractmethod
    def evaluateExchangeCost(self, elemIn, elemOut, sol):
        pass


