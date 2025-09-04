

class Solution:
    """
    Represents a solution in the GRASP framework.

    Stores a list of elements and its associated cost.
    """

    def __init__(self, other: "Solution" = None):
        if other is None:
            self.elements = []
            self.cost = float("inf")
        else:
            # Copy constructor
            self.elements = list(other.elements)
            self.cost = other.cost

    def add(self, elem):
        """Adds an element to the solution."""
        self.elements.append(elem)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self.elements)}], elements={self.elements}"

    def copy(self) -> "Solution":
        """Returns a copy of this solution."""
        return Solution(self)
