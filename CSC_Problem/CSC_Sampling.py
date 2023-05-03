import numpy as np

from pymoo.core.sampling import Sampling

from CSC_Problem import CSC_Problem

class CSC_Sampling(Sampling):

    def get_random_production_plan(self, problem: CSC_Problem):
        '''
        Returns a randomly generated production plan for the CSC problem, that respects the bounds of the market available materials.

        Parameters:
        -----------
        problem : CSC_Problem
            The problem instance for which the production plan should be generated.
        
        Returns:
        --------
        production_plan : np.array
            The randomly generated production plan.
        '''
        production_plan = np.zeros(problem.n_products)

        #create a random order to fill the genome
        genome_id_in_random_order = np.arange(problem.n_products)
        np.random.shuffle(genome_id_in_random_order)

        #calculate the market cap (max amount for each material that can be purchaced) for the current individual
        material_market_cap = problem.get_material_market_cap()

        for product_id in genome_id_in_random_order: #sample for gene / ampunt of products separately
            
            #calculate the maximum amount of product that could be produced
            product_cost = problem.Mc[product_id]
            limits = material_market_cap / product_cost
            limits = np.floor(limits)
            max_amount_of_product = min(limits)

            #generate the amount of products to be produced
            new_value = np.random.randint(0, max_amount_of_product + 1)

            #calculate the mew market cap for the next genome
            material_market_cap = material_market_cap - (product_cost * new_value)

            #set the value
            production_plan[product_id] = new_value

        return production_plan
        
    def get_ramdom_material_sourcing(self, problem: CSC_Problem, materials_needed):
        '''
        Returns a randomly generated material sourcing plan for a given production plan, respecting minimum and maximum order quantities at the suppliers.

        Parameters:
        -----------
        problem : CSC_Problem
            The problem instance for which the production plan should be generated.
        materials_needed : np.array
            Array containing for each material how much needs to be sourced for production.
        
        Returns:
        --------
        material_sourcing : np.array
            The randomly generated material sourcing plan.
        '''

        n_sources = problem.n_var - problem.n_products
        material_sourcing = np.zeros(n_sources)

        Cap = problem.Cap
        Moq = problem.Moq
        n_materials = problem.n_materials
        material_at_gene = problem.material_at_gene

        for material_id in range(0, n_materials): #sample for each material block separately
            current_block = np.where(material_at_gene == material_id+1)[0]
            if current_block.size != 0: #don't sample anythink if the block is not needed
                while materials_needed[material_id] > 0: #sample one block until the material need is met
                    
                    #choose a random supplier:
                    supplier_index_to_buy_from = np.random.choice(current_block)

                    #get what is still needed
                    current_need = materials_needed[material_id]                        


                    #calc maximum to buy
                    max_to_buy = min([ current_need, Cap[supplier_index_to_buy_from] - material_sourcing[supplier_index_to_buy_from]])
                    if max_to_buy <= 0: #start again if the maximum is already bought
                        continue
                    
                    
                    #calc the min to buy
                    min_to_buy = 1
                    if material_sourcing[supplier_index_to_buy_from] < Moq[supplier_index_to_buy_from]: #is we don't already order the Moq thats the new min
                        min_to_buy = Moq[supplier_index_to_buy_from]
                        
                    
                    #if we need less then the min, we have to purchace the min. Otherwise we randomly sample.
                    amount_to_buy = min_to_buy
                    if min_to_buy < max_to_buy:
                        amount_to_buy = np.random.randint(min_to_buy, max_to_buy + 1)
                    
                    material_sourcing[supplier_index_to_buy_from] += amount_to_buy
                    materials_needed[material_id] = current_need - amount_to_buy

        return material_sourcing


    def _do(self, problem : CSC_Problem, n_samples, **kwargs):

        X = np.zeros((n_samples, problem.n_var), dtype=int)
    
        for ind_id in range(n_samples):#sample each individual separately

            #generate random production plan
            production_plan = self.get_random_production_plan(problem)

            #generate random material sourcing
            materials_needed = (production_plan * problem.Mc).sum(axis=1)
            material_sourcing = self.get_ramdom_material_sourcing(problem, materials_needed)

            #set the value
            X[ind_id] = np.concatenate((material_sourcing, production_plan))

        return X