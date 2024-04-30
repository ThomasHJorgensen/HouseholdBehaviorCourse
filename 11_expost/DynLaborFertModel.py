import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class DynLaborFertModelClass(EconModelClass):

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

        par.beta_0 = 0.1 # weight on labor dis-utility (constant)
        par.beta_1 = 0.05 # additional weight on labor dis-utility (children)
        par.eta = -2.0 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.1 # labor income tax

        # children
        par.p_birth = 0.1
        par.util_child = 0.5

        # saving
        par.r = 0.02 # interest rate

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -10.0 # minimum point in wealth grid
        par.Na = 50 #70 # number of grid points in wealth grid 
        
        par.k_max = 20.0 # maximum point in wealth grid
        par.Nk = 20 #30 # number of grid points in wealth grid    

        par.Nn = 2 # number of children

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

        # c. number of children grid
        par.n_grid = np.arange(par.Nn)

        # d. solution arrays
        shape = (par.T,par.Nn,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # store the fertility effort-specific solutions
        shape_d = (par.T,2,par.Nn,par.Na,par.Nk)
        sol.c_d = np.nan + np.zeros(shape_d)
        sol.h_d = np.nan + np.zeros(shape_d)
        sol.V_d = np.nan + np.zeros(shape_d)

        # e. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.n = np.zeros(shape,dtype=np.int_)
        sim.birth = np.zeros(shape,dtype=np.int_)
        sim.effort = np.zeros(shape,dtype=np.int_)


        # f. draws used to simulate child arrival
        np.random.seed(9210)
        sim.draws_uniform = np.random.uniform(size=shape)

        # g. initialization
        sim.a_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.n_init = np.zeros(par.simN,dtype=np.int_)

        # h. vector of wages. Used for simulating elasticities
        par.w_vec = par.w * np.ones(par.T)


    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # b. solve last period
        
        # c. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: number of children, human capital and wealth in beginning of period
            for i_n,kids in enumerate(par.n_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_n,i_a,i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.

                        if t==par.T-1: # last period: no choice of fertility effort
                            
                            obj = lambda x: self.obj_last(x[0],assets,capital,kids)

                            # call optimizer
                            hours_min = np.fmax( - assets / self.wage_func(capital,t) + 1.0e-5 , 0.0) # minimum amount of hours that ensures positive consumption
                            init_h = np.maximum(hours_min,2.0) if i_a==0 else np.array([sol.h[t,i_n,i_a-1,i_k]])
                            res = minimize(obj,init_h,bounds=((hours_min,np.inf),),method='L-BFGS-B')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = -res.fun

                            # fill out discrete-choice specific values: Always choce not to concieve
                            idx_d = (t,0,i_n,i_a,i_k)
                            sol.c_d[idx_d] = sol.c[idx]
                            sol.h_d[idx_d] = sol.h[idx]
                            sol.V_d[idx_d] = sol.V[idx]

                            idx_d = (t,1,i_n,i_a,i_k)
                            sol.c_d[idx_d] = sol.c[idx]
                            sol.h_d[idx_d] = sol.h[idx]
                            sol.V_d[idx_d] = sol.V[idx] - 100000000.0

                        else:
                            
                            for i_e in (0,1):
                                effort = i_e

                                idx_d = (t,i_e,i_n,i_a,i_k)

                                # objective function: negative since we minimize
                                obj = lambda x: - self.value_of_choice(x[0],x[1],effort,assets,capital,kids,t)  

                                # bounds on consumption 
                                lb_c = 0.000001 # avoid dividing with zero
                                ub_c = np.inf

                                # bounds on hours
                                lb_h = 0.0
                                ub_h = np.inf 

                                bounds = ((lb_c,ub_c),(lb_h,ub_h))
                    
                                # call optimizer
                                idx_last = (t+1,i_e,i_n,i_a,i_k)
                                init = np.array([sol.c_d[idx_last],sol.h_d[idx_last]])
                                res = minimize(obj,init,bounds=bounds,method='L-BFGS-B',tol=1.0e-8) 
                            
                                # store results in choice-specific arrays
                                sol.c_d[idx_d] = res.x[0]
                                sol.h_d[idx_d] = res.x[1]
                                sol.V_d[idx_d] = -res.fun

                            # determine optimal fertility choice
                            idx = (t,i_n,i_a,i_k)
                            idx0 = (t,0,i_n,i_a,i_k)
                            idx1 = (t,1,i_n,i_a,i_k)
                            V0 = sol.V_d[idx0]
                            V1 = sol.V_d[idx1]

                            if V0>V1:
                                sol.c[idx] = sol.c_d[idx0]
                                sol.h[idx] = sol.h_d[idx0]
                                sol.V[idx] = V0
                            else:
                                sol.c[idx] = sol.c_d[idx1]
                                sol.h[idx] = sol.h_d[idx1]
                                sol.V[idx] = V1

    # last period
    def cons_last(self,hours,assets,capital):
        par = self.par

        income = self.wage_func(capital,par.T-1) * hours
        cons = assets + income
        return cons

    def obj_last(self,hours,assets,capital,kids):
        cons = self.cons_last(hours,assets,capital)
        return - self.util(cons,hours,kids)    

    # earlier periods
    def value_of_choice(self,cons,hours,effort,assets,capital,kids,t):

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
        util = self.util(cons,hours,kids)
        
        # d. *expected* continuation value from savings
        income = self.wage_func(capital,t) * hours
        a_next = (1.0+par.r)*(assets + income - cons)
        k_next = capital + hours

        # no birth
        kids_next = kids
        V_next = sol.V[t+1,kids_next]
        V_next_no_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

        if effort==0:
            EV_next = V_next_no_birth

        else:  
            # birth
            if (kids>=(par.Nn-1)):
                # cannot have more children
                V_next_birth = V_next_no_birth

            else:
                kids_next = kids + 1
                V_next = sol.V[t+1,kids_next]
                V_next_birth = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next)

            EV_next = par.p_birth * V_next_birth + (1-par.p_birth)*V_next_no_birth

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours,kids):
        par = self.par

        beta = par.beta_0 + par.beta_1*kids
        util_cons = (c)**(1.0+par.eta) / (1.0+par.eta)
        util_labor = - beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma)
        util_child = par.util_child * kids
        return  util_cons + util_labor + util_child

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
            sim.n[i,0] = sim.n_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # 0. optimal fertility effort choice
                idx0 = (t,0,sim.n[i,t])
                idx1 = (t,1,sim.n[i,t])
                V0 = interp_2d(par.a_grid,par.k_grid,sol.V_d[idx0],sim.a[i,t],sim.k[i,t])
                V1 = interp_2d(par.a_grid,par.k_grid,sol.V_d[idx1],sim.a[i,t],sim.k[i,t])
                if V0>V1:
                    sim.effort[i,t] = 0
                else:
                    sim.effort[i,t] = 1

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.effort[i,t],sim.n[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c_d[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h_d[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]

                    sim.birth[i,t] = 0 
                    if ((sim.effort[i,t]==1) & (sim.draws_uniform[i,t] <= par.p_birth) & (sim.n[i,t]<(par.Nn-1))):
                        sim.birth[i,t] = 1
                    sim.n[i,t+1] = sim.n[i,t] + sim.birth[i,t]
                    


