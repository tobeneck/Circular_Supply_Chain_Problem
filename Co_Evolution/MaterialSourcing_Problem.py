import numpy as np
from pymoo.core.problem import Problem

class MaterialSourcing_Problem(Problem):

    def __init__(self, n_materials, n_suppliers, n_products, Moq, Cap, Mp, Source_Type, materials_needed):
        '''
        Parameters:
        n_materials : int
            The number of materials in this problem.
        n_suppliers : int
            The number of suppliers in this problem.
        n_products : int
            The number of products that can be produced.
        Moq : numpy.array (2D)
            The minimum order quantity for each material (row) at each supplier (column).
        Cap : numpy.array (2D)
            The the production capacity, or maximum order quantity, for each material (row) at each supplier (column).
        Mp : numpy.array (2D)
            The price for each material (row) at each supplier (column).
        Type : numpy.array (2D)
            The material type for each material (row) at each supplier (column). False if virgin material, True if recycled material.
        materials_needed : numpy_array (2D)
            Vector containing for each material how much is needed in the current production plan.

        '''

        self.n_suppliers = n_suppliers
        self.n_materials = n_materials
        self.n_products = n_products

        self.Cap = np.ravel( Cap )
        cap_zero_indices = np.where(self.Cap == 0) #the indices where we the production capacity is zero
        self.Cap = np.delete(self.Cap, cap_zero_indices) #this contains obly the suppliers where material can be sourced from
        self.Moq = np.delete(np.ravel( Moq ), cap_zero_indices)
        self.Mp = np.delete(np.ravel( Mp ), cap_zero_indices)
        self.Source_Type = np.delete(np.ravel( Source_Type ), cap_zero_indices)
        
        self.materials_needed = materials_needed

        self.material_at_gene = np.array( [np.ones(n_suppliers)*(i+1) for i in range(n_materials)] ) #contains for each gene to which material it corresponds
        self.material_at_gene = np.delete( np.ravel( self.material_at_gene ), cap_zero_indices)
        
        self.supplier_at_gene = np.array( [np.arange(start=1, stop=n_suppliers+1) for i in range(n_materials)] ) #contains for each gene to which supplier it corresponds
        self.supplier_at_gene = np.delete( np.ravel( self.supplier_at_gene ), cap_zero_indices)

        print(self.material_at_gene)
        print(self.supplier_at_gene)
        print(self.Moq)
        print(self.Cap)
        print(self.Mp)
        

        xl = np.zeros( len(self.Cap) )
        xu = self.Cap

        super().__init__(n_var=np.count_nonzero(Cap),
                n_obj=2,
                xl=xl,
                xu=xu
                )
        


    def f2(self, x):
        '''
        Returns the second objective for the lower population, which is the ratio of virgin materials purchaced (to be minimized).

        Parameters:
        -----------
        x : np.array()
            The population to be evaluated.

        Returns:
        --------
        cost : np.array
            The cost for purchasing all materials.
        '''

        all_material_sourcing = x

        recycled_sources = np.where(self.Source_Type == True)
        virgin_material_sourcing = np.delete(all_material_sourcing, recycled_sources, axis=1)

        all_material = all_material_sourcing.sum(axis=1)
        virgin_material = virgin_material_sourcing.sum(axis=1)

        #replace all zero values with one to avoid dividing by zero. By replacing only the denominator the result will still be zero. Not buying any material is considered to be maximally sustainable.
        nothing_bought = np.where(all_material == 0)
        all_material[nothing_bought] = 1

        virgin_material_ratio = virgin_material / all_material

        return virgin_material_ratio

    def f1(self, x):
        '''
        Returns the first objective for the lower population, which is the cost of the purchace for all materials (to be minimized).

        Parameters:
        -----------
        x : np.array()
            The population to be evaluated.

        Returns:
        --------
        cost : np.array
            The cost for purchasing all materials.
        '''
        pop_size = len(x)

        #calculate cost
        Mp = np.full( (pop_size, len(self.Mp)), self.Mp )
        cost = x * Mp
        cost = cost.sum(axis=1)
        
        return cost
        



    def _evaluate(self, x, out, *args, **kwargs):
        f1 = self.f1(x)
        f2 = self.f2(x)
        out["F"] = np.column_stack([f1, f2])

