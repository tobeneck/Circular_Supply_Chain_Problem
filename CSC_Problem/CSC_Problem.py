import numpy as np
from pymoo.core.problem import Problem

class CSC(Problem):


    def get_material_market_cap(self):
        '''
        Returns for each material how much of it can be purchaced on the whole market (all suppliers).

        Parameters:
        -----------
        n_products : int
            The number of products that can be produced.
        Cap : np.array (2D)
            The the production capacity, or maximum order quantity, for each material (row) at each supplier (column).
        material_at_gene : numpy_array (2D)
            Array containing for each possible purchace in Cap to which material it belongs.

        Returns:
        --------
        material_market_cap : np.array
            Array containining for each material how much can be porchaced at the market.
        '''

        material_market_cap = np.zeros(self.n_products)

        for product_index in range(self.n_products):
            current_product_indices = np.where(self.material_at_gene == product_index + 1)[0]
            material_market_cap[product_index] = self.Cap[current_product_indices].sum()

        return material_market_cap
    
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

        self.Sp = Sp
        self.Mc = Mc

        self.material_at_gene = np.array( [np.ones(n_suppliers)*(i+1) for i in range(n_materials)] ) #contains for each gene to which material it corresponds
        self.material_at_gene = np.delete( np.ravel( self.material_at_gene ), cap_zero_indices)

        self.supplier_at_gene = np.array( [np.arange(start=1, stop=n_suppliers+1) for i in range(n_materials)] ) #contains for each gene to which supplier it corresponds
        self.supplier_at_gene = np.delete( np.ravel( self.supplier_at_gene ), cap_zero_indices)
        

        xl = np.zeros(len(self.Cap) + n_products )
        xu = self.Cap #he limits of each supplier
        xu = np.append(xu, self.xu_of_products(Cap, Mc)) #the theoretical maximum of each product that can be produced

        super().__init__(n_var=np.count_nonzero(Cap) + self.n_products,
                n_obj=2,
                xl=xl,
                xu=xu
                )
        


    def f2(self, x):
        '''
        Returns the second objective, modelling the sustainability (/percentage of recycled material used in production).
        Multiplied by -1 to make it a minimization problem.

        Parameters:
        -----------
        x : np.array()
            The population to be evaluated.

        Returns:
        --------
        virgin_material_ratio : np.array
            The ratio of virgin materials used in production. Should be minimized.
        '''

        all_material_sourcing = x[:,:len(self.Mp)]

        virgin__sources = np.where(self.Type == False)
        recycled_material_sourcing = np.delete(all_material_sourcing, virgin__sources, axis=1)

        all_material = all_material_sourcing.sum(axis=1)
        recycled_material = recycled_material_sourcing.sum(axis=1)

        #replace all zero values with ones to avoid dividing by zero. Not buying any material is considered to be maximally sustainable.
        nothing_bought = np.where(all_material == 0)
        all_material[nothing_bought] = 1
        recycled_material[nothing_bought] = 1

        recycled_material_ratio = recycled_material / all_material

        return recycled_material_ratio * -1

    def f1(self, x):
        '''
        Returns the first objective, modelling the profit.

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
        
        return (profit - cost) * -1
        



    def _evaluate(self, x, out, *args, **kwargs):
        f1 = self.f1(x)
        f2 = self.f2(x)
        out["F"] = np.column_stack([f1, f2])

