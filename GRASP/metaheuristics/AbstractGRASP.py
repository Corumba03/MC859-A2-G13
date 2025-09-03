__package__ = "metaheuristics.grasp"

from abc import ABC, abstractmethod
import random
import problems.Evaluator as Evaluator
import solutions.Solution as Solution

'''
Abstract class for metaheuristic GRASP (Greedy Randomized Adaptive Search
Procedure). It consider a minimization problem.
  
This Python port is based on the original Java implementation by 
ccavellucci and fusberti
'''

class AbstractGRASP(ABC):
    # flag that indicates whether the code should print more information on screen
    verbose = True

    # a random number generator
    random.seed(0)
    rng = random.random() 

    # The alpha parameter of GRASP
    _alpha = 0.0

    # the best (incumbet) solution cost
    _bestCost = float('inf')

    # the current solution cost
    _cost = float('inf')

    # the best solution
    _bestSol = Solution()

    # the current solution
    _sol = Solution()

    # the candidate List of elements to enter the solution
    _CL = []

    # the restricted candidate list
    _RCL = []

    '''
    Constructor for the AbstractGRASP class.

    Attributes:
    -----------
        objFunction (Evaluator): The objective function being optimized.
        alpha (float): The alpha parameter of GRASP, which controls the
                       greedyness-randomness balance.
        iterations (int): The number of iterations the GRASP main loop
                          executes.    
    '''
    def __init__(self, objFunction, iterations=1, alpha=0.0):
        self._ObjFunction = objFunction # The objettive being optimized
        self._alpha = alpha  # the alpha parameter of GRASP, which controls the greedyness-randomness balance.
        self._iterations = iterations # the number of iterations the GRASP main lopop executes.

    '''
    Creates the Candidate List, which is an ArrayList of candidate elements
    candidate elements that can enter a solution. The best candidates are
    efined through a quality threshold, delimited by the GRASP
    greedyness-randomness alpha parameter.

    Returns:
        The Restricted Candidate List (list).
    '''
    @abstractmethod
    def makeCL(self):
        pass

    '''
    Creates the Restricted Candidate List, which is an ArrayList of the best
    candidate elements that can enter a solution. The best candidates are
    defined through a quality threshold, delimited by the GRASP 
    greedyness-randomness alpha parameter.
    
    Returns:
        The Restricted Candidate List.
    '''
    @abstractmethod
    def makeRCL(self):
        pass

    '''
    Updates the Candidate List according to the current solution.
    In other words, this method is responsible for
    updating which elements are still viable to take part into the solution.
    '''
    @abstractmethod
    def updateCL(self): 
        pass

    '''
    Creates a new solution which is empty, i.e., does not contain any
    element.
    Returns:
        An empty solution.
    '''
    @abstractmethod
    def createEmptySol(self):
        pass

    '''
    The GRASP local search phase is responsible for repeatedly applying a
	neighborhood operation while the solution is getting improved, i.e.,
	until a local optimum is attained.

    Returns: 
	    A local optimum solution.
    '''
    @abstractmethod
    def localSearch(self):
        pass

    '''
    A standard stopping criteria for the constructive heuristic is to repeat
	until the current solution improves by inserting a new candidate
	element.
	
    Returns:
	    true if the criteria is met.
    '''
    def constructiveStoppingCriteria(self):
        return False if self._cost > self._sol.cost else True

    '''
    The GRASP constructive heuristic, which is responsible for building a
	feasible solution by selecting in a greedy-random fashion, candidate
	elements to enter the solution.
	
    Returns:
	    A feasible solution to the problem being minimized.
    '''
    def constructiveHeuristic(self):

        CL = self.makeCL()
        RCL = self.makeRCL()
        sol = self.createEmptySol()
        cost = float('inf')

        # Main loop, which repeats until the stopping criteria is reached.
        while not self.constructiveStoppingCriteria():
           
            maxCost = float('-inf')
            minCost = float('inf')
            cost = self._ObjFunction.evaluate(sol) # TODO implement evaluate
            self.updateCL()

            
            # Explore all candidate elements to enter the solution, saving the
            # highest and lowest cost variation achieved by the candidates.
            for c in CL:
                deltaCost = self._ObjFuction.evaluateInsertionCost(c, sol) # TODO implement evaluateInsertionCost
                if deltaCost < minCost:
                    minCost = deltaCost
                elif deltaCost > maxCost:
                        maxCost = deltaCost
                
            # Among all candidates, insert into the RCL those with the highest
            # performance using parameter alpha as threshold.
            for c in CL:
                deltaCost = self._ObjFuction.evaluateInsertionCost(c, sol) 
                if deltaCost <= minCost + self._alpha * (maxCost - minCost):
                    RCL.append(c)
            
            # Choose a candidate randomly from the RCL
            rndIndex = int(self.rng * len(RCL))
            inCand = RCL[rndIndex]
            CL.remove(inCand)
            sol.append(inCand)
            self._ObjFunction.evaluate(sol)
            RCL.clear()
        
        return sol
    
    '''
    The GRASP mainframe. It consists of a loop, in which each iteration goes
	through the constructive heuristic and local search. The best solution is
	returned as result.
	
    Returns:
	    The best feasible solution obtained throughout all iterations.
    '''
    def solve(self):
        
        bestSol = self.createEmptySol()
        for i in range(self._iterations):
           self.constructiveHeuristic()
           self.localSearch()
           if self._bestSol.cost > self._sol.cost:
               self._bestSol = self._sol
               self._bestCost = self._sol.cost
               if self.verbose:
                   print(f'(Iter. {i}) BestSol = {self._bestCost}')
        
        return self._bestSol