import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 10 # time periods
        
        # preferences
        par.rho = 0.98 # discount factor

        par.beta = 0.1 # weight on labor dis-utility
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 35.0 # maximum point in wealth grid
        par.Na = 50 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 25 # number of grid points in wealth grid      

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.a_grid = nonlinspace(0.0,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # c. solution arrays
        shape = (par.T,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # d. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)

        # e. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: human capital and wealth in beginning of period
            for ik,capital in enumerate(par.k_grid):
                for ia,assets in enumerate(par.a_grid):

                    # ii. find optimal consumption and hours at this level of wealth in this period t.

                    if t==par.T-1: # last period
                        obj = lambda x: self.obj_last(x[0],assets,capital)

                        constr = lambda x: self.cons_last(x[0],assets,capital)
                        nlc = NonlinearConstraint(constr, lb=0.0, ub=np.inf,keep_feasible=True)

                        # call optimizer
                        init_h = np.array([2.0]) if ia==0 else np.array([sol.h[t,ia-1,ik]]) # initial guess on optimal hours
                        res = minimize(obj,init_h,bounds=((0.0,np.inf),),constraints=nlc,method='trust-constr')

                        # store results
                        idx = (t,ia,ik)
                        sol.c[idx] = self.cons_last(res.x[0],assets,capital)
                        sol.h[idx] = res.x[0]
                        sol.V[idx] = -res.fun

                    else:

                        # objective function: negative since we minimize
                        obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,t)  

                        # bounds and constraint
                        # bounds on consumption 
                        lb_c = 0.000001 # avoid dividing with zero
                        ub_c = np.inf

                        # bounds on hours
                        lb_h = 0.0
                        ub_h = np.inf 

                        bounds = ((lb_c,ub_c),(lb_h,ub_h))
                        
                        # intertemporal budget constraint
                        savings = lambda x: assets + self.wage_func(capital) * x[1] - x[0]
                        nlc = NonlinearConstraint(savings, lb=0.0, ub=np.inf,keep_feasible=True)
            
                        # call optimizer
                        init = np.array([lb_c,1.0]) if ia==0 else np.array([sol.c[t,ia-1,ik],sol.h[t,ia-1,ik]]) # initial guess on optimal consumption and hours
                        res = minimize(obj,init,bounds=bounds,constraints=nlc,method='trust-constr')
                    
                        # store results
                        idx = (t,ia,ik)
                        sol.c[idx] = res.x[0]
                        sol.h[idx] = res.x[1]
                        sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital):
        par = self.par

        income = self.wage_func(capital) * hours
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital):
        cons = self.cons_last(hours,assets,capital)
        return - self.util(cons,hours)    

    # previous periods
    def value_of_choice(self,cons,hours,assets,capital,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. utility from consumption
        util = self.util(cons,hours)
        
        # c. continuation value from savings
        V_next = sol.V[t+1]
        income = self.wage_func(capital) * hours
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours
        V_next_interp = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        # d. return value of choice
        return util + par.beta*V_next_interp


    def util(self,c,h):
        par = self.par

        return (c)**(1.0+par.eta) / (1.0+par.eta) - par.beta*(h)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w * (1.0 + par.alpha * capital)

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
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):
                if t<par.T: # check that simulation does not go further than solution

                    # ii. interpolate optimal consumption and hours
                    sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[t],sim.a[i,t],sim.k[i,t])
                    sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[t],sim.a[i,t],sim.k[i,t])

                    # iii. store next-period states
                    if t<par.simT-1:
                        income = self.wage_func(sim.k[i,t])*sim.h[i,t]
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                        sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]


