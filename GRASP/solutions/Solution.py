__package__ = "solutions"

class Solution(list):
    
    def __init__(self, sol=None):
        super().__init__(sol if sol is not None else [])
        self.cost = float('inf')
        if sol is not None and hasattr(sol, 'cost'):
            self.cost = sol.cost

    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self)}], elements={super().__str__()}"