import numpy as np

from CSC_Problem.CSC_Problem import CSC
from CSC_Problem.CSC_Sampling import CSC_Sampling

#definition of the problem instance
F = 50
n_materials = 2
n_products = 2
n_suppliers = 3

Moq = [[15, 2, 1],
        [10, 3, 2]]
Cap = [[100, 20, 30],
        [70, 30, 50]]
Mp  = [[2, 2.5, 3],
        [2, 2.5, 2.5]]
Type  = [[False, False, True],
            [False, False, True]]
Mc = [[3, 1],
        [1, 3]]
Sp = [15, 15]

Moq = np.ravel( np.array(Moq) )
Cap = np.array(Cap)
Mp = np.array(Mp)
Mc = np.array(Mc)
Sp = np.array(Sp)

problem = CSC(n_materials,
                n_suppliers,
                n_products,
                Moq,
                Cap,
                Mp,
                Type,
                Mc,
                Sp,
                F
                )

#sample population
sampling = CSC_Sampling()

n_samples = 10
pop = sampling._do(problem, n_samples)


#evaluate the population
fitness = problem.evaluate(pop)
print(pop)
print(fitness * -1) #*-1 as pymoo only accepts minimization problems, so the fitness needs to be inverted.