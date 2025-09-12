import random
from GRASP.metaheuristics.AbstractGRASP import AbstractGRASP as GRASP
from GRASP.Solution import Solution
from GRASP.problems import Evaluator

class RandomGreedyGRASP(GRASP):
    '''
    Implements a GRASP with a hybrid constructive phase and a Best-Improving local search.
    - The first 'p' steps are purely random.
    - The subsequent steps are semi-greedy, controlled by 'alpha'.
    '''
    
    def __init__(
        self, 
        obj_function: Evaluator,
        p: int,          # Número de passos puramente aleatórios
        alpha: float,    # Fator de aleatoriedade para os passos semi-gulosos
        iterations: int = 1,
        maximize: bool = True
    ):
        super().__init__(obj_function, alpha=alpha, iterations=iterations, maximize=maximize)
        
        if p < 0:
            raise ValueError("O parâmetro 'p' não pode ser negativo.")
        self.p = p

    def constructive_heuristic(self) -> Solution:
        """
        Builds a solution using the hybrid "Random plus (Semi-)Greedy" scheme.
        """
        self.sol = self.create_empty_sol()
        self.CL = self.make_CL()
        
        # --- FASE 1: PASSOS TOTALMENTE ALEATÓRIOS ---
        num_random_steps = min(self.p, len(self.CL))
        for _ in range(num_random_steps):
            if not self.CL:
                break
            
            chosen_element = random.choice(list(self.CL))
            
            self.sol.add(chosen_element)
            self.CL.discard(chosen_element)
            self.update_CL()

        # --- FASE 2: PASSOS SEMI-GULOSOS (usando alpha) ---
        while self.CL:
            # Constrói a RCL usando a lógica do alpha
            self.make_RCL()
            
            if not self.RCL:
                break # Encerra se a RCL estiver vazia
            
            # Sorteia um elemento da RCL (e não mais da CL inteira)
            chosen_element = random.choice(list(self.RCL))
            
            self.sol.add(chosen_element)
            self.CL.discard(chosen_element)
            self.update_CL()
            
        return self.sol

    def make_RCL(self) -> set[int]:
        """
        Return a subset of CL based on cost ranking and self.alpha.
        Esta função agora é essencial para a Fase 2.
        """
        if not self.CL:
            self.RCL = set()
            return self.RCL

        deltas = {elem: self.obj_function.evaluate_insertion_cost(elem, self.sol) for elem in self.CL}
        
        # Pode acontecer de não haver mais movimentos válidos
        if not deltas:
            self.RCL = set()
            return self.RCL
            
        min_cost = min(deltas.values())
        max_cost = max(deltas.values())
        
        if self.maximize:
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta >= threshold}
        else:
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta <= threshold}

        # Garante que a RCL nunca esteja vazia se a CL não estiver
        if not self.RCL and self.CL:
            # Se o alpha for muito restritivo, adiciona pelo menos o melhor candidato
            if self.maximize:
                best_element = max(deltas, key=deltas.get)
            else:
                best_element = min(deltas, key=deltas.get)
            self.RCL = {best_element}

        return self.RCL

    def local_search(self, sol: Solution) -> Solution:
        """
        Applies best-improvement local search on the given solution.
        (Este método permanece exatamente o mesmo da versão anterior)
        """
        current_best_sol = sol.copy()
        self.obj_function.evaluate(current_best_sol)

        while True:
            best_neighbor_in_iteration = current_best_sol
            
            # Remoção, Troca e Inserção... (código omitido por brevidade, é o mesmo de antes)
            # --- 1. EXPLORAÇÃO COMPLETA: REMOÇÃO ---
            for elem_out in list(current_best_sol.elements):
                neighbor = current_best_sol.remove(elem_out)
                self.obj_function.evaluate(neighbor)
                if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                    best_neighbor_in_iteration = neighbor

            # --- 2. EXPLORAÇÃO COMPLETA: TROCA (EXCHANGE) ---
            for elem_out in list(current_best_sol.elements):
                for elem_in in range(self.obj_function.get_domain_size()):
                    if elem_in not in current_best_sol.elements:
                        neighbor = current_best_sol.exchange(elem_in, elem_out)
                        self.obj_function.evaluate(neighbor)
                        if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                            best_neighbor_in_iteration = neighbor
            
            # --- 3. EXPLORAÇÃO COMPLETA: INSERÇÃO ---
            for elem_in in range(self.obj_function.get_domain_size()):
                if elem_in not in current_best_sol.elements:
                    neighbor = current_best_sol.insert(elem_in)
                    self.obj_function.evaluate(neighbor)
                    if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                        best_neighbor_in_iteration = neighbor
            
            # --- DECISÃO FINAL DA ITERAÇÃO ---
            if best_neighbor_in_iteration is current_best_sol:
                break
            else:
                current_best_sol = best_neighbor_in_iteration.copy()
                
        return current_best_sol

    def solve(self):
        """
        Executes the GRASP loop.
        (Este método também permanece o mesmo)
        """
        self.best_sol = self.create_empty_sol()
        self.obj_function.evaluate(self.best_sol)
        best_solutions_per_iter = []

        for i in range(self.iterations):
            constructed_sol = self.constructive_heuristic()
            local_opt_sol = self.local_search(constructed_sol)
            
            if self.is_improvement(local_opt_sol.cost, self.best_sol.cost):
                self.best_sol = local_opt_sol.copy()
            
            best_solutions_per_iter.append(self.best_sol.copy())
            print(f"(Iter. {i+1}) BestSol = {self.best_sol}")

        return best_solutions_per_iter

    # --- MÉTODOS ABSTRATOS IMPLEMENTADOS ---

    def create_empty_sol(self) -> Solution:
        """Cria e retorna uma solução vazia."""
        return Solution(maximize=self.maximize)

    def make_CL(self) -> set[int]:
        """Cria a lista de candidatos inicial (todos os elementos)."""
        return set(range(self.obj_function.get_domain_size()))

    def update_CL(self):
        """Atualiza a lista de candidatos, se necessário."""
        # Para muitos problemas, como o seu, pode não ser necessário fazer nada aqui.
        pass

    # --- RESTANTE DOS SEUS MÉTODOS ---

    def is_improvement(self, new_cost: float, current_cost: float) -> bool:
            """
            Determina se o novo custo é uma melhora sobre o custo atual.
            Leva em conta se o problema é de maximização ou minimização.
            """
            if self.maximize:
                return new_cost > current_cost
            else:
                return new_cost < current_cost

    def select_alpha(self):
        """
        Select an alpha from the pool according to the current probabilities.
        """
        # ATENÇÃO: Corrigindo a chamada com k=1 que discutimos antes
        self.alpha = choices(self.alpha_pool, weights=self.probabilities, k=1)[0]
        return self.alpha
           
    # ... (make_RCL, local_search, solve, etc. continuam aqui) ...
    def make_RCL(self) -> set[int]:
        """
        Return a subset of CL based on cost ranking and self.alpha.
        Esta função agora é essencial para a Fase 2.
        """
        if not self.CL:
            self.RCL = set()
            return self.RCL

        deltas = {elem: self.obj_function.evaluate_insertion_cost(elem, self.sol) for elem in self.CL}
        
        # Pode acontecer de não haver mais movimentos válidos
        if not deltas:
            self.RCL = set()
            return self.RCL
            
        min_cost = min(deltas.values())
        max_cost = max(deltas.values())
        
        if self.maximize:
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta >= threshold}
        else:
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            self.RCL = {c for c, delta in deltas.items() if delta <= threshold}

        # Garante que a RCL nunca esteja vazia se a CL não estiver
        if not self.RCL and self.CL:
            # Se o alpha for muito restritivo, adiciona pelo menos o melhor candidato
            if self.maximize:
                best_element = max(deltas, key=deltas.get)
            else:
                best_element = min(deltas, key=deltas.get)
            self.RCL = {best_element}

        return self.RCL

    def local_search(self, sol: Solution) -> Solution:
        """
        Applies best-improvement local search on the given solution.
        (Este método permanece exatamente o mesmo da versão anterior)
        """
        current_best_sol = sol.copy()
        self.obj_function.evaluate(current_best_sol)

        while True:
            best_neighbor_in_iteration = current_best_sol
            
            # Remoção, Troca e Inserção... (código omitido por brevidade, é o mesmo de antes)
            # --- 1. EXPLORAÇÃO COMPLETA: REMOÇÃO ---
            for elem_out in list(current_best_sol.elements):
                neighbor = current_best_sol.remove(elem_out)
                self.obj_function.evaluate(neighbor)
                if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                    best_neighbor_in_iteration = neighbor

            # --- 2. EXPLORAÇÃO COMPLETA: TROCA (EXCHANGE) ---
            for elem_out in list(current_best_sol.elements):
                for elem_in in range(self.obj_function.get_domain_size()):
                    if elem_in not in current_best_sol.elements:
                        neighbor = current_best_sol.exchange(elem_in, elem_out)
                        self.obj_function.evaluate(neighbor)
                        if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                            best_neighbor_in_iteration = neighbor
            
            # --- 3. EXPLORAÇÃO COMPLETA: INSERÇÃO ---
            for elem_in in range(self.obj_function.get_domain_size()):
                if elem_in not in current_best_sol.elements:
                    neighbor = current_best_sol.insert(elem_in)
                    self.obj_function.evaluate(neighbor)
                    if self.is_improvement(neighbor.cost, best_neighbor_in_iteration.cost):
                        best_neighbor_in_iteration = neighbor
            
            # --- DECISÃO FINAL DA ITERAÇÃO ---
            if best_neighbor_in_iteration is current_best_sol:
                break
            else:
                current_best_sol = best_neighbor_in_iteration.copy()
                
        return current_best_sol

    def solve(self):
        """
        Executes the GRASP loop.
        (Este método também permanece o mesmo)
        """
        self.best_sol = self.create_empty_sol()
        self.obj_function.evaluate(self.best_sol)
        best_solutions_per_iter = []

        for i in range(self.iterations):
            constructed_sol = self.constructive_heuristic()
            local_opt_sol = self.local_search(constructed_sol)
            
            if self.is_improvement(local_opt_sol.cost, self.best_sol.cost):
                self.best_sol = local_opt_sol.copy()
            
            best_solutions_per_iter.append(self.best_sol.copy())
            print(f"(Iter. {i+1}) BestSol = {self.best_sol}")

        return best_solutions_per_iter