class SetCover:
    """
    Generic Set Cover manager.
    Each variable x_i is associated with a set S_i of elements it covers.
    Provides methods to check feasibility and filter candidates.
    """

    def __init__(self, sets: list[list[int]]):
        """
        :param sets: list of sets S_i, where each S_i is a list of elements covered by variable i
        """
        self.sets = sets
        # Determine total elements that need to be covered
        self.num_elements = max((max(s) for s in sets if s), default=-1) + 1

    def is_feasible(self, solution: list[int]) -> bool:
        """
        Checks if a solution covers all required elements.
        :param solution: list of variable indices included in the solution
        :return: True if feasible, False otherwise
        """
        covered = set()
        for i in solution:
            covered.update(self.sets[i])
        return len(covered) == self.num_elements

    def feasible_candidates(self, solution: list[int], candidates: list[int]) -> list[int]:
        """
        Filters a list of candidate variables and returns only those that
        can be added without violating the coverage rules.
        Currently a simple version that just returns all candidates.
        Can be extended with more sophisticated feasibility checks.
        :param solution: current solution
        :param candidates: potential candidates to add
        :return: list of feasible candidates
        """
        return candidates

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
