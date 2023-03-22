import numpy as np

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2

from MaterialSourcing_Problem import MaterialSourcing_Problem
from MaterialSourcing_Sampling import MaterialSourcing_Sampling
from MaterialSourcing_Crosover import MaterialSourcing_Crossover
from MaterialSourcing_Mutation import MaterialSourcing_Mutation


F = 50
n_materials = 2
n_products = 2
n_suppliers = 3

Moq = [[15, 2, 1],
        [10, 3, 2]]
Cap = [[100, 20, 0],
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


materials_needed = [50, 50]

problem = MaterialSourcing_Problem(n_materials,
                n_suppliers,
                n_products,
                Moq,
                Cap,
                Mp,
                Type,
                materials_needed
                )




#sampler = Sampling_Lower()


algorithm = NSGA2(pop_size=10,
                  sampling=MaterialSourcing_Sampling(),
                  crossover=MaterialSourcing_Crossover(material_blocks = problem.material_at_gene),
                  mutation=MaterialSourcing_Mutation(material_blocks = problem.material_at_gene),
                )

res = minimize(problem,
               algorithm,
               ('n_gen', 20),
               seed=1,
               verbose=False)


print(res.algorithm.pop.get("X"))
print(res.algorithm.pop.get("F"))
exit()
#dummy population
x = [
    [0,1,2,3,4 ,5],
    [6,7,8,9,10,11],
    [0,0,0,0,0 ,0]
]
x = np.array(x)

print(problem.f1(x))
print(problem.f2(x))