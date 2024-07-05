import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

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
        par.rho = 0.99 # discount factor

        par.beta = 1.0 # weight on labor dis-utility
        par.eta = -2.5 # CRRA coefficient
        par.gamma = 2.5 # curvature on labor hours 

        # income
        par.alpha = 0.1 # human capital accumulation 
        par.w = 1.0 # wage base level
        par.tau = 0.12 # labor income tax

        # saving
        par.r = 0.03 # interest rate
        par.tau_a = 0.0 # wealth tax

        # human capital
        par.depre = 0.2 # depreciation rate
        par.prob_depre = 0.6 # probability of depreciation

        # disability
        par.prob_d = 0.1 
        par.benefit = 0.2
        par.disable_scale = 0.5

        # grids
        par.a_max = 5.0 # maximum point in wealth grid
        par.a_min = -5.0 # minimum point in wealth grid
        par.Na = 40 # number of grid points in wealth grid 
        
        par.k_max = 15.0 # maximum point in wealth grid
        par.Nk = 10 # number of grid points in wealth grid   

        par.num_d = 1 # disability states

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

        # unemployment grid (zero is employed -> num_u=1 and prob_u=0.0 gives baseline)
        par.d_grid = np.arange(par.num_d)
        
        # c. solution arrays
        shape = (par.T,par.num_d,par.Na,par.Nk)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # d. simulation arrays
        shape = (par.simN,par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.d = np.zeros(shape,dtype=np.int_)

        sim.budget = np.nan + np.zeros(shape)
        sim.util = np.nan + np.zeros(shape)

        # e. initialization
        np.random.seed(3500)
        sim.a_init = np.fmax(0.0 , np.random.normal(size=par.simN))
        sim.k_init = np.fmax(0.0 , 0.5*np.random.normal(size=par.simN))
        sim.d_init = np.zeros(par.simN,dtype=np.int_)

        # f.uniform draws
        sim.draw_uniform = np.random.uniform(0.0,1.0,(par.simN,par.simT))
        sim.draw_uniform_d = np.random.uniform(0.0,1.0,(par.simN,par.simT))

    ############
    # Solution #
    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol
        
        # a. loop backwards (over all periods)
        for t in reversed(range(par.T)):

            # i. loop over state variables: unemployment, human capital and wealth in beginning of period
            for i_d,disabled in enumerate(par.d_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_k,capital in enumerate(par.k_grid):
                        idx = (t,i_d,i_a,i_k)
                        idx_last = (t+1,i_d,i_a,i_k)
                        idx_prev_asset = (t,i_d,i_a-1,i_k)

                        # ii. find optimal consumption and hours at this level of wealth in this period t.
                        if t==par.T-1: # last period
  
                            obj = lambda x: self.obj_last(x[0],assets,capital,disabled)

                            # call optimizer
                            hours_min = np.fmax( - assets / self.wage_func(capital,disabled) + 1.0e-5 , 0.0) # minimum amount of hours that ensures positive consumption
                            init_h = np.maximum(hours_min,2.0) if i_a==0 else np.array([sol.h[idx_prev_asset]])
                            res = minimize(obj,init_h,bounds=((hours_min,np.inf),),method='L-BFGS-B')

                            # store results
                            sol.c[idx] = self.cons_last(res.x[0],assets,capital,disabled)
                            sol.h[idx] = res.x[0]
                            sol.V[idx] = self.util(sol.c[idx],sol.h[idx])

                        else:
                            
                            # objective function: negative since we minimize
                            obj = lambda x: - self.value_of_choice(x[0],x[1],assets,capital,disabled,t)  

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = np.inf

                            # bounds on hours
                            lb_h = 0.0
                            ub_h = np.inf 

                            bounds = ((lb_c,ub_c),(lb_h,ub_h))
                
                            # call optimizer
                            init = np.array([sol.c[idx_last],sol.h[idx_last]])
                            res = minimize(obj,init,bounds=bounds,method='L-BFGS-B',tol=1.0e-10) 
                        
                            # store results
                            sol.c[idx] = res.x[0]
                            sol.h[idx] = res.x[1]
                            sol.V[idx] = -res.fun

    # last period
    def cons_last(self,hours,assets,capital,disabled):
        income = self.wage_func(capital,disabled) * hours
        benefits = self.par.benefit*disabled
        wealth_tax_rate = self.par.tau_a * (assets>0.0)
        cons = (1-wealth_tax_rate)*assets + income + benefits
        return cons

    def obj_last(self,hours,assets,capital,disabled):
        cons = self.cons_last(hours,assets,capital,disabled)
        return - self.util(cons,hours) 

    # earlier periods
    def value_of_choice(self,cons,hours,assets,capital,disabled,t):

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
        util = self.util(cons,hours)
        
        
        # d. continuation value from savings
        EV_next = 0.0
        for d_next in par.d_grid:
            if par.num_d==1:
                prob_d = 1.0
            else:
                prob_d = par.prob_d*(1+disabled)
                if d_next==0:
                    prob_d = 1.0 - prob_d

            V_next = sol.V[t+1,d_next]
            a_next = self.wealth_trans(assets,capital,disabled,hours,cons)
            
            # e. expected value wrt human capital depreciation
            k_next_no = capital + hours
            V_next_no = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next_no)

            k_next_depre = (1-par.depre)*capital + hours
            V_next_depre = interp_2d(par.a_grid,par.k_grid,V_next,a_next,k_next_depre)

            EV_next += prob_d * (par.prob_depre*V_next_depre + (1-par.prob_depre)*V_next_no)

        # e. return value of choice (including penalty)
        return util + par.rho*EV_next + penalty


    def util(self,c,hours):
        par = self.par

        return (c)**(1.0+par.eta) / (1.0+par.eta) - par.beta*(hours)**(1.0+par.gamma) / (1.0+par.gamma) 

    def wage_func_before_tax(self,capital,disabled):
        par = self.par
        scale = (1-par.disable_scale) if disabled==1 else 1.0
        return scale * par.w * (1.0 + par.alpha * capital)

    def wage_func(self,capital,disabled):
        # after tax wage rate
        return (1.0 - self.par.tau )*self.wage_func_before_tax(capital,disabled)
    
    def wealth_trans(self,assets,capital,disabled,hours,cons):
        par = self.par

        income_after_tax = self.wage_func(capital,disabled) * hours
        wealth_tax_rate = par.tau_a * (assets>0.0)
        benefits = par.benefit * disabled
        a_next = (1.0+par.r)*((1.0-wealth_tax_rate)*assets + income_after_tax + benefits - cons)

        return a_next

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
            sim.d[i,0] = sim.d_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                idx_sol = (t,sim.d[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.k_grid,sol.c[idx_sol],sim.a[i,t],sim.k[i,t])
                sim.h[i,t] = interp_2d(par.a_grid,par.k_grid,sol.h[idx_sol],sim.a[i,t],sim.k[i,t])

                # iii. save taxes and utility
                labor_tax = par.tau * self.wage_func_before_tax(sim.k[i,t],sim.d[i,t])*sim.h[i,t]
                wealth_tax = par.tau_a * sim.a[i,t] * (sim.a[i,t]>0.0)
                disability_benefits = sim.d[i,t] * par.benefit
                sim.budget[i,t] = labor_tax + wealth_tax - disability_benefits

                sim.util[i,t] = self.util(sim.c[i,t],sim.h[i,t])

                # iv. store next-period states
                if t<par.simT-1:
                    sim.a[i,t+1] = self.wealth_trans(sim.a[i,t],sim.k[i,t],sim.d[i,t],sim.h[i,t],sim.c[i,t])
                    
                    # human capital depreciation
                    depre = 0.0
                    if sim.draw_uniform[i,t] <= par.prob_depre:
                        depre = par.depre
                    sim.k[i,t+1] = (1-depre)*sim.k[i,t] + sim.h[i,t]

                    # disability
                    disabled_next = 0
                    if par.num_d>1:
                        prob = par.prob_d * (1.0 + sim.d[i,t])
                        if sim.draw_uniform_d[i,t] <= prob:
                            disabled_next = 1
                    sim.d[i,t+1] = disabled_next
                    


