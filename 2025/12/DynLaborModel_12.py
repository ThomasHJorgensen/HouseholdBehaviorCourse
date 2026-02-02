import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d, interp_1d

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
        par.beta_2 = 0.2 # weight on labor dis-utility for type 2
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # saving
        par.r = 0.02 # interest rate

        # health risk
        par.p = 0.2 # probability of human capital destruction

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 30 # number of grid points in wealth grid    

        par.Nm = 2 # Number of types

        # simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals


    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T
        
        # a. asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1.1)

        # b. human capital grid
        par.k_grid = nonlinspace(0.0,par.k_max,par.Nk,1.1)

        # types
        par.m_grid = np.arange(1,par.Nm+1) # types

        # c. solution arrays
        shape = (par.T,par.Nm,par.Na,par.Nk)
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

        # f. random uniform draws to simulate human capital destruction
        np.random.seed(9210) # for reproducibility
        sim.draws_uni = np.random.uniform(size=(par.simN,par.simT))

        # types 
        sim.m = np.random.randint(1,par.Nm+1,size=par.simN) # random types for each individual

        # f. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)


    ############
    # Solution #
    def solve(self,do_accurate=False):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: human capital and wealth in beginning of period
            for i_m,m_type in enumerate(par.m_grid): # new state related to types of workers
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_m,i_a,i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.

                        if t==par.T-1: # last period
                            obj = lambda x: self.obj_last(x[0],assets,capital,m_type)

                            # call optimizer
                            hours_min = np.fmax( - assets / self.wage_func(capital,t) + 1.0e-5 , 0.0) # minimum amount of hours that ensures positive consumption
                            init_h = np.maximum(hours_min,2.0) if i_a==0 else np.array([sol.h[t,i_m,i_a-1,i_k]])
                            res = minimize(obj,init_h,bounds=((hours_min,np.inf),),method='L-BFGS-B')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                        else:
                            
                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,m_type,t)  

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_c,ub_c),(lb_h,ub_h))
                
                            # call optimizer
                            init = np.array([lb_c,1.0]) if ((i_a==0) & (i_k==0)) else res.x  # initial guess on optimal consumption and hours: last found optimal values
                            res = minimize(obj,init,bounds=bounds,method='L-BFGS-B',tol=1.0e-10)

                            # store results
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital):
        par = self.par

        income = self.wage_func(capital,par.T-1) * hours
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,m_type):
        cons = self.cons_last(hours,assets,capital)
        return - self.util(cons,hours,m_type) # negative since we minimize

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,m_type,t):

        # a. unpack
        par = self.par
        sol = self.sol

        # b. penalty for violating bounds. 
        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5
        if hours < 0.0:
            penalty += hours*1_000.0
            hours = 0.0

        # c. utility from consumption
        util = self.util(cons,hours,m_type)
        
        # d. continuation value from savings
        V_next = sol.V[t+1,m_type-1] # m_type-1 is the index associated with each type as m_type is (1,2) and index starts at zero in Python.
        income = self.wage_func(capital,t) * hours
        a_next = (1.0+par.r)*(assets + income - cons)
        
        # calculate expected value of next period
        k_next_no_shock = capital + hours
        V_next_interp_no_shock = interp_2d(par.a_grid,par.k_grid, V_next,        a_next,k_next_no_shock)

        if par.p>0.0:
            k_next_shock = 0.0 # human capital is destroyed
            V_next_interp_shock = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next_shock)
            # similar (but potentially faster):
            # V_next_interp_shock = interp_1d(par.a_grid,          sol.V[t+1,m_type-1,:,0], a_next)

            EV_next = par.p*V_next_interp_shock + (1.0-par.p)*V_next_interp_no_shock

        else:
            EV_next = V_next_interp_no_shock

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,m_type):
        par = self.par

        beta = par.beta
        if m_type==2:
            beta = par.beta_2

        return (c)**(1.0+par.eta) / (1.0+par.eta) - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func(self,capital,t):
        # after tax wage rate
        par = self.par

        return (1.0 - par.tau )* par.w_vec[t] * (1.0 + par.alpha * capital)

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

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.m[i]-1) # m_type-1 is the index associated with each type as m_type is (1,2) and index starts at zero in Python.
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])

                    # introduce shock to human capital
                    if sim.draws_uni[i,t] <= par.p: # using the fact stated in the notebook
                        sim.k[i,t+1] = 0.0
                    else:
                        sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]


