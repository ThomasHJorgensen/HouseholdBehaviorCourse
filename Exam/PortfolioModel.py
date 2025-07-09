import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_2d
from consav.quadrature import normal_gauss_hermite

class PortfolioModelClass(EconModelClass):

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
        par.rho = 2.0 # CRRA coefficient

        # income
        par.y = 1.0 # income level
        
        # interest rates
        par.Ra = 1.01 # risk-free interest rate
        par.mu = 0.0 # mean of risky asset return shocks
        par.mu0 = 1.03 # mean of initial risky asset return shocks
        par.sigma2 = 0.03 # variance of risky asset return shocks

        # grids
        par.s_max = 8.0 # maximum point in savings grid
        par.Ns = 25 # number of grid points in savings grid. Very low to make it feasable.    

        par.Rb_max = 1.5 # maximum point in risky asset return grid
        par.NRb = 16 # number of grid points in risky asset return grid. Very low to make it feasable.

        par.Neps = 5 # number of points in risky asset return shocks

        # simulation
        par.seed = 9210
        par.simT = par.T # number of periods
        par.simN = 5_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim
        
        # a. asset grid
        par.s_grid = nonlinspace(0.0,par.s_max,par.Ns,1.1)
        
        # b. risky asset return grid
        Rb_grid_lower = np.linspace(0.0,par.Ra,par.NRb//2)
        Rb_grid_upper = nonlinspace(par.Ra*1.01,par.Rb_max,par.NRb-par.NRb//2,1.1)
        par.Rb_grid = np.array([Rb_grid_lower,Rb_grid_upper]).ravel()

        # c. risky asset return shock grid (Normal)
        if par.sigma2>0.0 and par.Neps>0:
            par.eps_grid,par.eps_weight = normal_gauss_hermite(np.sqrt(par.sigma2),par.Neps)
        else:
            par.eps_grid,par.eps_weight = np.array([0.0]), np.array([1.0])
            par.Neps = 1
            par.sigma2 = 0.0

        # d. solution arrays
        shape = (par.T,par.Ns,par.NRb)
        sol.c = np.nan + np.zeros(shape)
        sol.x = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.x = np.nan + np.zeros(shape)
        
        sim.a = np.nan + np.zeros(shape)
        sim.b = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.Rb = np.nan + np.zeros(shape)
        
        sim.resources = np.nan + np.zeros(shape)
        
        # f. random log-normal mean one income shocks
        np.random.seed(par.seed)
        sim.eps = par.mu + np.sqrt(par.sigma2)*np.random.normal(size=shape)

        # g. initialization
        sim.init_s = 0.5 + np.random.uniform(size=par.simN)
        sim.init_Rb = par.mu0 + np.zeros(par.simN)

        
    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        t = par.T-1
        for i_Rb,Rb in enumerate(par.Rb_grid):
            sol.c[t,:,i_Rb] = par.s_grid + par.y # consumption in last period is equal to savings + income
            sol.x[t,:,i_Rb] = np.zeros(par.s_grid.shape) # does not matter in the final period->set tp zero
            sol.V[t,:,i_Rb] = self.util(sol.c[t,:,i_Rb])

        # c. loop backwards
        for t in reversed(range(par.T-1)):

            # i. loop over state variable: resources in beginning of period
            for i_s,savings in enumerate(par.s_grid):
                for i_Rb,Rb in enumerate(par.Rb_grid):
                    idx = (t,i_s,i_Rb) # index for current period, savings and risky asset return
                    idx_last = (t+1,i_s,i_Rb) # index for next period, savings and risky asset return

                    # ii. find optimal consumption and portfolio allocation at this level of savings and risky asset return

                    # objective function: negative since we minimize
                    obj = lambda choice: - self.value_of_choice(choice[0],choice[1],savings,Rb,t)  

                    # bounds on consumption
                    lb_c = 0.00001 # avoid dividing with zero
                    ub_c = savings + par.y # upper bound on consumption
                    
                    # bounds on portfolio allocation share in risky asset
                    lb_x = 0.0
                    ub_x = 1.0

                    bounds = ((lb_c,ub_c),(lb_x,ub_x))

                    # call optimizer
                    init = np.array([0.5*ub_c , 0.5]) if t==(par.T-2) else np.array([sol.c[idx_last],sol.x[idx_last]])# initial guess on optimal consumption
                    res = minimize(obj,init,bounds=bounds)
                    opt_c = res.x[0]
                    opt_x = res.x[1]

                    # check corner cases in portfolio allocation
                    obj0 = lambda cons: - self.value_of_choice(cons[0],0.0,savings,Rb,t)
                    obj1 = lambda cons: - self.value_of_choice(cons[0],1.0,savings,Rb,t)

                    init = np.array([opt_c])
                    res0 = minimize(obj0,init,bounds=((lb_c,ub_c),))
                    res1 = minimize(obj1,init,bounds=((lb_c,ub_c),))
                    if res0.fun<res.fun:
                        res = res0
                        opt_c = res.x[0]
                        opt_x = 0.0
                    if res1.fun<res.fun:
                        res = res1
                        opt_c = res.x[0]
                        opt_x = 1.0

                    # store results
                    sol.c[idx] = opt_c
                    sol.x[idx] = opt_x
                    sol.V[idx] = -res.fun
        

    def value_of_choice(self,cons,x_share,savings,Rb,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # bounds and penalty
        penalty = 0.0
        if x_share<0.0:
            penalty += 1000.0*x_share
            x_share = 0.0
        if x_share>1.0:
            penalty += 1000.0*(1.0-x_share)
            x_share = 1.0
        if cons<=0.00001:
            penalty += -1000.0*abs(0.00001-cons)
            cons = 0.00001
        if cons> savings + par.y:
            penalty += 1000.0*(savings + par.y - cons)
            cons = savings + par.y


        # b. utility from consumption
        util = self.util(cons)
        
        # c. expected continuation value from savings
        resources = savings + par.y - cons

        # d. share in riskless and risky asset
        a_next = self.a_next_func(resources,x_share) 
        
        # loop over risky asset return shocks
        EV_next = 0.0
        for i_eps,eps in enumerate(par.eps_grid):
            prob_eps = par.eps_weight[i_eps]
            
            # risky asset return next period
            Rb_next = self.Rb_next_func(Rb,eps) 
            b_next = self.b_next_func(resources,x_share,Rb_next)

            # savings next period
            savings_next = a_next + b_next

            # interpolate next period value function for this combination of 
            V_next = sol.V[t+1]        
            V_next_interp = interp_2d(par.s_grid,par.Rb_grid,V_next,savings_next,Rb_next)

            # weight the interpolated value with the likelihood
            EV_next += prob_eps * V_next_interp

        # d. return value of choice
        return util + par.beta*EV_next + penalty


    def util(self,c):
        par = self.par

        return (c)**(1.0-par.rho) / (1.0-par.rho)

    def a_next_func(self,resources,x_share):
        # risk-free asset next period
        par = self.par

        a_share = (1.0-x_share) * resources
        a_next = par.Ra * a_share
        return a_next

    def b_next_func(self,resources,x_share,Rb_next):
        # risky asset next period
        par = self.par

        b_share = x_share * resources
        b_next = Rb_next * b_share
        return b_next
    
    def Rb_next_func(self,Rb,eps):
        # risky asset return next period
        par = self.par

        Rb_next = np.fmax(Rb + eps , 0.0)
        return Rb_next

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
            sim.s[i,0] = sim.init_s[i]
            sim.Rb[i,0] = sim.init_Rb[i]

            for t in range(par.simT):
                if t<par.T: # check that simulation does not go further than solution                 

                    # iii. interpolate optimal consumption and portfolio allocation
                    sim.c[i,t] = interp_2d(par.s_grid,par.Rb_grid,sol.c[t],sim.s[i,t],sim.Rb[i,t])
                    sim.c[i,t] = np.fmax(sim.c[i,t],0.0) # avoid negative consumption
                    
                    sim.x[i,t] = interp_2d(par.s_grid,par.Rb_grid,sol.x[t],sim.s[i,t],sim.Rb[i,t])
                    sim.x[i,t] = np.clip(sim.x[i,t],0.0,1.0) # keep share in between 0 and 1

                    # resources (w in the model)
                    sim.resources[i,t] = sim.s[i,t] + par.y - sim.c[i,t]

                    # iv. Update next-period states
                    if t<par.simT-1:
                        sim.Rb[i,t+1] = np.fmin(par.Rb_grid[-1],self.Rb_next_func(sim.Rb[i,t],sim.eps[i,t+1]) ) # cap at maximum in grid for stability

                        sim.a[i,t+1] = self.a_next_func(sim.resources[i,t],sim.x[i,t])  # risk-free asset
                        sim.b[i,t+1] = self.b_next_func(sim.resources[i,t],sim.x[i,t],sim.Rb[i,t+1]) # risky asset
                        
                        sim.s[i,t+1] = sim.a[i,t+1] + sim.b[i,t+1] # total savings next period
                        