class SetCover():
    """
    Generic Set Cover manager.
    Provides methods to check feasibility and filter candidates.
    """

    def __init__(self, sets: list[set[int]], num_elements: int):
        self.sets = sets
        self.num_elements = num_elements
    
    def getdomainSize(self) -> int:
        return len(self.sets)

    def is_feasible(self, solution: set[int]) -> bool:
        """
        Checks if a solution covers all required elements.
        :param solution: list of sets selected (by their indices)
        :return: True if feasible, False otherwise
        """
        covered = set()
        for i in solution: # This builds the union of the sets in the solution
            covered.update(self.sets[i])
        return len(covered) == self.num_elements

    def coverage(self, solution: list[int]) -> set[int]:
        """
        Returns the set of elements covered by the current solution.
        :param solution: list of variable indices
        :return: set of covered elements
        """
        covered = set()
        for i in solution:
            covered.update(self.sets[i])
        return covered
