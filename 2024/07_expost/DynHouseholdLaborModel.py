import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynHouseholdLaborModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.beta = 0.98 # discount factor

        par.rho_const_1 = 0.05
        par.rho_kids_1 = 0.01
        par.rho_const_2 = 0.05
        par.rho_kids_2 = 0.01

        par.eta = -1.5 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # children
        par.p_birth = 0.1

        # income
        par.wage_const_1 = np.log(10_000.0) # constant, men
        par.wage_const_2 = np.log(10_000.0) # constant, women
        par.wage_K_1 = 0.1 # return on human capital, men
        par.wage_K_2 = 0.1 # return on human capital, women

        par.delta = 0.1 # depreciation in human capital

        # taxes
        par.tax_scale = 2.278029 # from Borella et al. (2023), singles: 1.765038
        par.tax_pow = 0.0861765 # from Borella et al. (2023), singles: 0.0646416

        # child-related transfers
        par.uncon_uni = 0.1
        par.means_level = 0.1
        par.means_slope = 0.001
        par.cond = -0.1
        par.cond_high = -0.1

        # grids        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid    

        par.num_n = 2 # maximum number in my grid over children

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        # reform
        par.joint_tax = True


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # b. number of children grid
        par.n_grid = np.arange(0,par.num_n)

        # d. solution arrays
        shape = (par.T,par.num_n,par.Nk,par.Nk)
        sol.h1 = np.nan + np.zeros(shape)
        sol.h2 = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.h1 = np.nan + np.zeros(shape)
        sim.h2 = np.nan + np.zeros(shape)
        sim.k1 = np.nan + np.zeros(shape)
        sim.k2 = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        
        sim.income1 = np.nan + np.zeros(shape)
        sim.income2 = np.nan + np.zeros(shape)

        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.k1_init = np.zeros(par.simN)
        sim.k2_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: human capital for each household member
            for i_n,kids in enumerate(par.n_grid):
                for i_k1,capital1 in enumerate(par.k_grid):
                    for i_k2,capital2 in enumerate(par.k_grid):
                        idx = (t,i_n,i_k1,i_k2)
                        
                        # ii. find optimal hours of both members at this level of wealth in this period t.
                        if t==(par.T-1): # last period
                            obj = lambda x: -self.util(x[0],x[1],kids,capital1,capital2)

                        else:
                            obj = lambda x: - self.value_of_choice(x[0],x[1],t,kids,capital1,capital2)  

                        # call optimizer
                        bounds = [(0,np.inf) for i in range(2)]
                        
                        init_h = np.array([0.1,0.1])
                        if i_k1>0: 
                            init_h[0] = sol.h1[t,i_n,i_k1-1,i_k2]
                        if i_k2>0: 
                            init_h[1] = sol.h2[t,i_n,i_k1,i_k2-1]

                        res = minimize(obj,init_h,bounds=bounds) 

                        # store results
                        sol.h1[idx] = res.x[0]
                        sol.h2[idx] = res.x[1]
                        sol.V[idx] = -res.fun
 

    def value_of_choice(self,hours1,hours2,t,kids,capital1,capital2):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. current utility
        util = self.util(hours1,hours2,kids,capital1,capital2)
        
        # c. continuation value
        k1_next = (1.0-par.delta)*capital1 + hours1
        k2_next = (1.0-par.delta)*capital2 + hours2

        # no birth
        kids_next = kids  
        V_next = sol.V[t+1,kids_next]     
        V_next_no_birth = interp_2d(par.k_grid,par.k_grid,V_next,k1_next,k2_next)

        # birth
        if (kids>=(par.num_n-1)):
            # cannot have more children
            V_next_birth = V_next_no_birth

        else:
            kids_next = kids + 1
            V_next = sol.V[t+1,kids_next]
            V_next_birth = interp_2d(par.k_grid,par.k_grid,V_next,k1_next,k2_next)

        EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth


        # d. return value of choice
        return util + par.beta*EV_next


    # relevant functions
    def consumption(self,hours1,hours2,kids,capital1,capital2):
        par = self.par

        income1 = self.wage_func(capital1,1) * hours1
        income2 = self.wage_func(capital2,2) * hours2
        income_hh = income1+income2

        trans = self.child_tran(hours1,hours2,income_hh,kids)

        total_income = income_hh + trans

        tax = self.tax_func(total_income)
        
        return total_income - tax

    def wage_func(self,capital,sex):
        # before tax wage rate
        par = self.par

        constant = par.wage_const_1
        return_K = par.wage_K_1
        if sex>1:
            constant = par.wage_const_2
            return_K = par.wage_K_2

        return np.exp(constant + return_K * capital)

    def tax_func(self,income):
        par = self.par

        rate = 1.0 - par.tax_scale*(income**(-par.tax_pow))
        return rate*income

    def child_tran(self,hours1,hours2,income_hh,kids):
        par = self.par
        if kids<1:
            return 0.0
        
        else:
            C1 = par.uncon_uni                           #unconditional, universal transfer (>0)
            C2 = np.fmax(par.means_level - par.means_slope*income_hh , 0.0) #means-tested transfer (>0)
            # child-care related (net-of-subsidy costs)
            both_work = (hours1>0) * (hours2>0)
            C3 = par.cond*both_work                      #all working couples has this net cost (<0)
            C4 = par.cond_high*both_work*(income_hh>0.5) #low-income couples do not have this net-cost (<0)

        return C1+C2+C3+C4

    def util(self,hours1,hours2,kids,capital1,capital2):
        par = self.par

        cons = self.consumption(hours1,hours2,kids,capital1,capital2)

        rho_1 = par.rho_const_1 + par.rho_kids_1*kids
        rho_2 = par.rho_const_2 + par.rho_kids_2*kids

        util_cons = 2*(cons/2)**(1.0+par.eta) / (1.0+par.eta)
        util_hours1 = rho_1*(hours1)**(1.0+par.gamma) / (1.0+par.gamma)
        util_hours2 = rho_2*(hours2)**(1.0+par.gamma) / (1.0+par.gamma)

        return util_cons - util_hours1 - util_hours2

    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.k1[i,0] = sim.k1_init[i]
            sim.k2[i,0] = sim.k2_init[i]
            sim.n[i,0] = sim.n_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal hours
                idx_sol = (t,sim.n[i,t])
                sim.h1[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h1[idx_sol],sim.k1[i,t],sim.k2[i,t])
                sim.h2[i,t] = interp_2d(par.k_grid,par.k_grid,sol.h2[idx_sol],sim.k1[i,t],sim.k2[i,t])

                # store income
                sim.income1[i,t] = self.wage_func(sim.k1[i,t],1)*sim.h1[i,t]
                sim.income2[i,t] = self.wage_func(sim.k2[i,t],2)*sim.h2[i,t]

                # iii. store next-period states
                if t<par.simT-1:
                    sim.k1[i,t+1] = (1.0-par.delta)*sim.k1[i,t] + sim.h1[i,t]
                    sim.k2[i,t+1] = (1.0-par.delta)*sim.k2[i,t] + sim.h2[i,t]

                    birth = 0 
                    if ((sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.num_n-1))):
                        birth = 1
                    sim.n[i,t+1] = sim.n[i,t] + birth
                    


