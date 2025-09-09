from GRASP import Solution
from GRASP.metaheuristics import AbstractGRASP, ReactiveGRASP
from GRASP.problems import Evaluator, QBF, SC_QBF, SetCover


def main():
    
    # Read the number of variables (n)
    n = int(input())

    # Read the number of elements in each set (but not used afterwards)
    _ = list(map(int, input().split()))

    # Initialize list to hold the sets
    sets = []

    # Read n sets of integers, one per line
    for _ in range(n):
        sets.append(set(map(int, input().split())))

    # Initialize list to hold the coefficient matrix
    A = []

    # Read n rows of the coefficient matrix (expected to be upper triangular)
    for line in range(n):
        line = list(map(int, input().split()))
        A.append(line)

    # Model creation

    solver = ReactiveGRASP(
        obj_function = SC_QBF(n, A, sets),
        iterations=1
    )

    solution = solver.solve()
    print(f"{solution}")


if __name__=='__main__':
    main()
