import numpy as np
from pymoo.core.problem import Problem

class CSCP(Problem): #Circular Supply Chain Optimization Problem

    def xu_of_products(self, Cap, Mc):
        '''
        Returns per product the theoretical maximum that could be produced (if no other product would be pdoruced).
        Needs tp be called after self.n_products and self.n_materials are defined.

        Parameters:
        -----------
        Cap : numpy.array (2D)
            The the production capacity, or maximum order quantity, for each material (row) at each supplier (column).
        Mc : numpy.array (2D)
            The material (column) cost of each product (row).
        
        Returns:
        --------
        xu_products : numpy.array
            The theoretical maximum amount of each product that could be produced with all resources on the market.
        '''
        xu_products = []

        market_cap = np.ravel( np.sum(Cap, axis=1) )
        for p_index in range(self.n_products):
            curr_p_m_cost = Mc[p_index]
            current_upper_limit = np.iinfo(np.int64).max
            for m_index in range(self.n_materials): #the cost for each material
                curr_m_cost = curr_p_m_cost[m_index]
                production_limit_p_m = np.floor( curr_m_cost / market_cap[m_index] )#the production limit only concerning the current material
                if curr_m_cost != 0 and production_limit_p_m < current_upper_limit:
                    current_upper_limit = production_limit_p_m
            xu_products.append(current_upper_limit)
        
        return xu_products

    def get_absolute_material_values(self, x):
        '''
        Parameters:
        -----------
        x : numpy_array
            The population for which the absolute material needs need to be returned.
        
        Returns:
        --------
        abs_x : numpy_array
            The population with absolute purchace values instead of relative values.
        '''
        #TODO: calculate the absolute meterial need for a product plan


        #TODO 1. calculate a material needs vector from the amount of product neccecary

        pass
    
    def get_relative_material_values(self):
        #TODO: calculate the relative buying values of materials
        pass

    def __init__(self, n_materials, n_suppliers, n_products, Moq, Cap, Mp, Type, Mc, Sp, F):
        '''
        TODO

        Parameters:
        n_materials : int
            The number of materials in this problem.
        n_suppliers : int
            The number of suppliers in this problem.
        n_products : int
            The number of products in this problem
        Moq : numpy.array (2D)
            The minimum order quantity for each material (row) at each supplier (column).
        Cap : numpy.array (2D)
            The the production capacity, or maximum order quantity, for each material (row) at each supplier (column).
        Mp : numpy.array (2D)
            The price for each material (row) at each supplier (column).
        Type : numpy.array (2D)
            The material type for each material (row) at each supplier (column). False if virgin material, True if recycled material.
        Mc : numpy.array (2D)
            The material (column) cost of each product (row).
        Sp : numpy.array
            The sale price of each product.
        F : number
            The fixed cost.

        '''

        self.F = F
        self.n_suppliers = n_suppliers
        self.n_materials = n_materials
        self.n_products = n_products

        self.Cap = np.ravel( Cap )
        cap_zero_indices = np.where(self.Cap == 0) #the indices where we the production capacity is zero
        self.Cap = np.delete(self.Cap, cap_zero_indices) #this contains obly the suppliers where material can be sourced from
        self.Moq = np.delete(np.ravel( Moq ), cap_zero_indices)
        self.Mp = np.delete(np.ravel( Mp ), cap_zero_indices)
        self.Type = np.delete(np.ravel( Type ), cap_zero_indices)
        self.Mc = np.delete(np.ravel( Mc ), cap_zero_indices)

        self.Sp = Sp
        

        xl = np.zeros(len(self.Cap) + n_products )
        xu = self.Cap
        xu = np.append(xu, self.xu_of_products(Cap, Mc))

        super().__init__(n_var=np.count_nonzero(Cap) + self.n_products,
                n_obj=2,
                xl=xl,
                xu=xu
                )
        


    def f2(self, x):

        all_material_sourcing = x[:,:len(self.Mp)]

        virgin_sources = np.where(self.Type == False)
        virgin_material_sourcing = np.delete(all_material_sourcing, virgin_sources, axis=1)

        all_material = all_material_sourcing.sum(axis=1)
        virgin_material = virgin_material_sourcing.sum(axis=1)

        #replace all zero values with ones to avoid dividing by zero. Not buying any material is considered to be maximally sustainable.
        nothing_bought = np.where(all_material == 0)
        all_material[nothing_bought] = 1
        virgin_material[nothing_bought] = 1

        virgin_material_ratio = virgin_material / all_material

        return virgin_material_ratio

    def f1(self, x):
        '''
        Returns the first objective, the profit.

        Parameters:
        -----------
        x : np.array()
            The population to be evaluated.

        Returns:
        --------
        income : np.array
            The income generated from selling the product.
        '''
        pop_size = len(x)

        #calculate profit
        products = x[:,-self.n_products:]
        Sp = np.full( (pop_size, self.n_products) , self.Sp)
        profit = products * Sp
        profit = profit.sum(axis=1)
        print(profit)

        #calculate cost
        material_sourcing = x[:,:len(self.Mp)]
        Mp = np.full( (pop_size, len(self.Mp)), self.Mp )
        cost = material_sourcing * Mp
        cost = cost.sum(axis=1)
        print(cost)
        
        return profit - cost
        



    def _evaluate(self, x, out, *args, **kwargs):
        #TODO: calc f_1
        f1 = self.f1(x)
        #TODO: calc f_2
        f2 = (x[:, 0]-1)**2 + x[:, 1]**2
        out["F"] = np.column_stack([f1, f2])

