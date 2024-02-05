import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d

class ConSavModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 20 # time periods
        
        # preferences
        par.beta = 0.98 # discount factor
        par.rho = 2.0 # CRRA coefficient

        # income
        par.y = 1.0 # income level

        # saving
        par.r = 0.02 # interest rate

        # grid
        par.a_max = 30.0 # maximum point in wealth grid
        par.Na = 200 # number of grid points in wealth grid      

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

        # b. income
        par.yt = par.y * np.ones(par.T)

        # c. solution arrays
        shape = (par.T,par.Na)
        sol.c = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # d. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)

        # e. initialization
        sim.a_init = np.zeros(par.simN)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        t = par.T-1
        sol.c[t,:] = par.a_grid + par.yt[t]
        sol.V[t,:] = self.util(sol.c[t,:])

        # c. loop backwards [note, the last element, N, in range(N) is not included in the loop due to index starting at 0]
        for t in reversed(range(par.T-1)):

            # i. loop over state varible: wealth in beginning of period
            for ia,assets in enumerate(par.a_grid):

                # ii. find optimal consumption at this level of wealth in this period t.

                # objective function: negative since we minimize
                obj = lambda c: - self.value_of_choice(c[0],assets,t)  

                # bounds on consumption
                lb = 0.000001 # avoid dividing with zero
                ub = assets + par.yt[t] 

                # call optimizer
                if lb>=ub: # if the bounds are not feasible, set consumption to all resources 
                    sol.c[t,ia] = ub
                    sol.V[t,ia] = -obj(sol.c[t,ia])

                else:
                    c_init = np.array(lb+0.5*ub) # initial guess on optimal consumption
                    res = minimize(obj,c_init,bounds=((lb,ub),),method='SLSQP')
                    
                    # store results
                    sol.c[t,ia] = res.x[0]
                    sol.V[t,ia] = -res.fun
        

    def value_of_choice(self,cons,assets,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. utility from consumption
        util = self.util(cons)
        
        # c. continuation value from savings
        V_next = sol.V[t+1]
        a_next = (1.0+par.r)*(assets + par.yt[t] - cons)
        V_next_interp = interp_1d(par.a_grid,V_next,a_next)

        # d. return value of choice
        return util + par.beta*V_next_interp


    def util(self,c):
        par = self.par

        return (c)**(1.0-par.rho) / (1.0-par.rho)


    ##############
    # Simulation #
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize assets
            sim.a[i,0] = sim.a_init[i]

            for t in range(par.simT):
                if t<par.T: # check that simulation does not go further than solution

                    # ii. interpolate optimal consumption
                    sim.c[i,t] = interp_1d(par.a_grid,sol.c[t],sim.a[i,t])

                    # iii. store savings (next-period state)
                    if t<par.simT-1:
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + par.yt[t] - sim.c[i,t])


