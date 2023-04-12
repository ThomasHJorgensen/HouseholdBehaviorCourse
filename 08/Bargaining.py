import numpy as np
import scipy.optimize as optimize

from EconModel import EconModelClass
from consav.grids import nonlinspace
from consav import linear_interp, linear_interp_1d
from consav import quadrature

# user-specified functions
import UserFunctions as usr

# set gender indication as globals
woman = 1
man = 2

class HouseholdModelClass(EconModelClass):
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. cpp
        self.cpp_filename = 'cppfuncs/solve.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        par = self.par
        
        par.R = 1.03
        par.beta = 1.0/par.R # Discount factor
        
        par.div_A_share = 0.5 # divorce share of wealth to wife
        
        # income
        par.inc_w = 1.0
        par.inc_m = 1.0

        # Utility: gender-specific parameters
        par.rho_w = 2.0        # CRRA
        par.rho_m = 2.0        # CRRA
        
        par.alpha1_w = 1.0
        par.alpha1_m = 1.0
        
        par.alpha2_w = 1.0
        par.alpha2_m = 1.0
        
        par.phi_w = 0.2
        par.phi_m = 0.2
        
        # state variables
        par.T = 10
        
        # wealth
        par.num_A = 50
        par.max_A = 5.0
        
        # bargaining power
        par.num_power = 21

        # love/match quality
        par.num_love = 41
        par.max_love = 1.0

        par.sigma_love = 0.1
        par.num_shock_love = 5

        # pre-computation
        par.num_Ctot = 100
        par.max_Ctot = par.max_A*2

        par.num_A_pd = par.num_A * 2

        # simulation
        par.seed = 9210
        par.simT = par.T
        par.simN = 50_000
        
    def allocate(self):
        par = self.par
        sol = self.sol
        sim = self.sim

        # setup grids
        par.simT = par.T
        self.setup_grids()
        
        # singles
        shape_single = (par.T,par.num_A)
        sol.Vw_single = np.nan + np.ones(shape_single)
        sol.Vm_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_single = np.nan + np.ones(shape_single)

        sol.Vw_trans_single = np.nan + np.ones(shape_single)
        sol.Vm_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_priv_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_pub_trans_single = np.nan + np.ones(shape_single)
        sol.Cw_tot_trans_single = np.nan + np.ones(shape_single)
        sol.Cm_tot_trans_single = np.nan + np.ones(shape_single)

        # couples
        shape_couple = (par.T,par.num_power,par.num_love,par.num_A)
        sol.Vw_couple = np.nan + np.ones(shape_couple)
        sol.Vm_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_couple = np.nan + np.ones(shape_couple)

        sol.Vw_remain_couple = np.nan + np.ones(shape_couple)
        sol.Vm_remain_couple = np.nan + np.ones(shape_couple)
        
        sol.Cw_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.Cm_priv_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_pub_remain_couple = np.nan + np.ones(shape_couple)
        sol.C_tot_remain_couple = np.nan + np.ones(shape_couple)

        sol.power_idx = np.zeros(shape_couple,dtype=np.int_)
        sol.power = np.zeros(shape_couple)

        # temporary containers
        sol.savings_vec = np.zeros(par.num_shock_love)
        sol.Vw_plus_vec = np.zeros(par.num_shock_love) 
        sol.Vm_plus_vec = np.zeros(par.num_shock_love) 

        # EGM
        sol.marg_V_couple = np.zeros(shape_couple)
        sol.marg_V_remain_couple = np.zeros(shape_couple)

        shape_egm = (par.num_power,par.num_love,par.num_A_pd)
        sol.EmargU_pd = np.zeros(shape_egm)
        sol.C_tot_pd = np.zeros(shape_egm)
        sol.M_pd = np.zeros(shape_egm)

        # pre-compute optimal consumption allocation
        shape_pre = (par.num_power,par.num_Ctot)
        sol.pre_Ctot_Cw_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_Cm_priv = np.nan + np.ones(shape_pre)
        sol.pre_Ctot_C_pub = np.nan + np.ones(shape_pre)
        
        # simulation
        shape_sim = (par.simN,par.simT)
        sim.Cw_priv = np.nan + np.ones(shape_sim)
        sim.Cm_priv = np.nan + np.ones(shape_sim)
        sim.Cw_pub = np.nan + np.ones(shape_sim)
        sim.Cm_pub = np.nan + np.ones(shape_sim)
        sim.Cw_tot = np.nan + np.ones(shape_sim)
        sim.Cm_tot = np.nan + np.ones(shape_sim)
        sim.C_tot = np.nan + np.ones(shape_sim)
        
        sim.A = np.nan + np.ones(shape_sim)
        sim.Aw = np.nan + np.ones(shape_sim)
        sim.Am = np.nan + np.ones(shape_sim)
        sim.couple = np.nan + np.ones(shape_sim)
        sim.power_idx = np.ones(shape_sim,dtype=np.int_)
        sim.power = np.nan + np.ones(shape_sim)
        sim.love = np.nan + np.ones(shape_sim)

        # shocks
        np.random.seed(par.seed)
        sim.draw_love = np.random.normal(size=shape_sim)

        # initial distribution
        sim.init_A = par.grid_A[0] + np.zeros(par.simN)
        sim.init_Aw = np.zeros(par.simN)
        sim.init_Am = np.zeros(par.simN)
        sim.init_couple = np.ones(par.simN,dtype=np.bool)
        sim.init_power_idx = par.num_power//2 * np.ones(par.simN,dtype=np.int_)
        sim.init_love = np.zeros(par.simN)
        
    def setup_grids(self):
        par = self.par
        
        # wealth. Single grids are such to avoid interpolation
        par.grid_A = nonlinspace(0.0,par.max_A,par.num_A,1.1)

        par.grid_Aw = par.div_A_share * par.grid_A
        par.grid_Am = (1.0 - par.div_A_share) * par.grid_A

        # power. non-linear grid with more mass in both tails.
        odd_num = np.mod(par.num_power,2)
        first_part = nonlinspace(0.0,0.5,(par.num_power+odd_num)//2,1.3)
        last_part = np.flip(1.0 - nonlinspace(0.0,0.5,(par.num_power-odd_num)//2 + 1,1.3))[1:]
        par.grid_power = np.append(first_part,last_part)

        # love grid and shock
        if par.num_love>1:
            par.grid_love = np.linspace(-par.max_love,par.max_love,par.num_love)
        else:
            par.grid_love = np.array([0.0])

        if par.sigma_love<=1.0e-6:
            par.num_shock_love = 1
            par.grid_shock_love,par.grid_weight_love = np.array([0.0]),np.array([1.0])

        else:
            par.grid_shock_love,par.grid_weight_love = quadrature.normal_gauss_hermite(par.sigma_love,par.num_shock_love)

        # pre-computation
        par.grid_Ctot = nonlinspace(1.0e-6,par.max_Ctot,par.num_Ctot,1.1)

        # EGM
        par.grid_util = np.nan + np.ones((par.num_power,par.num_Ctot))
        par.grid_marg_u = np.nan + np.ones(par.grid_util.shape)
        par.grid_inv_marg_u = np.flip(par.grid_Ctot)
        par.grid_marg_u_for_inv = np.nan + np.ones(par.grid_util.shape)

        par.grid_A_pd = nonlinspace(0.0,par.max_A,par.num_A_pd,1.1)

    def solve(self):
        sol = self.sol
        par = self.par 

        # setup grids
        self.setup_grids()

        # precompute the optimal intra-temporal consumption allocation for couples given total consumpotion
        for iP,power in enumerate(par.grid_power):
            for i,C_tot in enumerate(par.grid_Ctot):
                sol.pre_Ctot_Cw_priv[iP,i], sol.pre_Ctot_Cm_priv[iP,i], sol.pre_Ctot_C_pub[iP,i] = solve_intraperiod_couple(C_tot,power,par)

        # loop backwards
        for t in reversed(range(par.T)):
            self.solve_single(t)
            self.solve_couple(t)

        # total consumption
        sol.C_tot_couple = sol.Cw_priv_couple + sol.Cm_priv_couple + sol.C_pub_couple
        sol.C_tot_remain_couple = sol.Cw_priv_remain_couple + sol.Cm_priv_remain_couple + sol.C_pub_remain_couple
        sol.Cw_tot_single = sol.Cw_priv_single + sol.Cw_pub_single
        sol.Cm_tot_single = sol.Cm_priv_single + sol.Cm_pub_single

        # value of transitioning to singlehood. Done here because absorbing . it is the same as entering period as single.
        sol.Vw_trans_single = sol.Vw_single.copy()
        sol.Vm_trans_single = sol.Vm_single.copy()
        sol.Cw_priv_trans_single = sol.Cw_priv_single.copy()
        sol.Cm_priv_trans_single = sol.Cm_priv_single.copy()
        sol.Cw_pub_trans_single = sol.Cw_pub_single.copy()
        sol.Cm_pub_trans_single = sol.Cm_pub_single.copy()
        sol.Cw_tot_trans_single = sol.Cw_tot_single.copy()
        sol.Cm_tot_trans_single = sol.Cm_tot_single.copy()
                      
    def solve_single(self,t):
        par = self.par
        sol = self.sol
        
        # loop through state variable: wealth
        for iA in range(par.num_A):
            idx = (t,iA)

            # resources
            Aw = par.grid_Aw[iA]
            Am = par.grid_Am[iA]

            Mw = usr.resources_single(Aw,woman,par) 
            Mm = usr.resources_single(Am,man,par) 

            if t == (par.T-1): # terminal period
                
                # intra-period allocation: consume all resources
                sol.Cw_priv_single[idx],sol.Cw_pub_single[idx] = intraperiod_allocation_single(Mw,woman,par)
                sol.Vw_single[idx] = usr.util(sol.Cw_priv_single[idx],sol.Cw_pub_single[idx],woman,par)
                
                sol.Cm_priv_single[idx],sol.Cm_pub_single[idx] = intraperiod_allocation_single(Mm,man,par)
                sol.Vm_single[idx] = usr.util(sol.Cm_priv_single[idx],sol.Cm_pub_single[idx],man,par)
            
            else: # earlier periods

                # search over optimal total consumption, C
                obj_w = lambda C_tot: - self.value_of_choice_single(C_tot[0],Mw,woman,sol.Vw_single[t+1])
                obj_m = lambda C_tot: - self.value_of_choice_single(C_tot[0],Mm,man,sol.Vm_single[t+1])
                
                res_w = optimize.minimize(obj_w,Mw/2.0,bounds=((1.0e-8,Mw),))
                res_m = optimize.minimize(obj_m,Mm/2.0,bounds=((1.0e-8,Mm),))
                
                # store results
                Cw = res_w.x
                sol.Cw_priv_single[idx],sol.Cw_pub_single[idx] = intraperiod_allocation_single(Cw,woman,par)
                sol.Vw_single[idx] = -res_w.fun
                
                Cm = res_m.x
                sol.Cm_priv_single[idx],sol.Cm_pub_single[idx] = intraperiod_allocation_single(Cm,man,par)
                sol.Vm_single[idx] = -res_m.fun                
                
    def solve_couple(self,t):
        par = self.par
        sol = self.sol

        remain_Vw,remain_Vm,remain_Cw_priv,remain_Cm_priv,remain_C_pub = np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power),np.ones(par.num_power)
        
        Vw_next = None
        Vm_next = None
        for iL,love in enumerate(par.grid_love):
            for iA,A in enumerate(par.grid_A):
                M_resources = usr.resources_couple(A,par) 
                
                starting_val = None
                for iP,power in enumerate(par.grid_power):
                    # continuation values
                    if t<(par.T-1):
                        Vw_next = self.sol.Vw_couple[t+1,iP]
                        Vm_next = self.sol.Vm_couple[t+1,iP]

                    # starting values
                    if iP>0:
                        C_tot_last = remain_Cw_priv[iP-1] + remain_Cm_priv[iP-1] + remain_C_pub[iP-1]
                        starting_val = np.array([C_tot_last])
                    
                    # solve problem if remining married
                    remain_Cw_priv[iP], remain_Cm_priv[iP], remain_C_pub[iP], remain_Vw[iP], remain_Vm[iP] = self.solve_remain_couple(t,M_resources,iL,iP,power,Vw_next,Vm_next,starting_val=starting_val)

                # check the participation constraints
                idx_single = (t,iA)
                idx_couple = lambda iP: (t,iP,iL,iA)

                list_start_as_couple = (sol.Vw_couple,sol.Vm_couple,sol.Cw_priv_couple,sol.Cm_priv_couple,sol.C_pub_couple)
                list_remain_couple = (remain_Vw,remain_Vm,remain_Cw_priv,remain_Cm_priv,remain_C_pub)
                list_trans_to_single = (sol.Vw_single,sol.Vm_single,sol.Cw_priv_single,sol.Cm_priv_single,sol.Cw_pub_single) # last input here not important in case of divorce
                
                Sw = remain_Vw - sol.Vw_single[idx_single] 
                Sm = remain_Vm - sol.Vm_single[idx_single] 
                
                check_participation_constraints(sol.power_idx,sol.power,Sw,Sm,idx_single,idx_couple,list_start_as_couple,list_remain_couple,list_trans_to_single, par)

                # save remain values
                for iP,power in enumerate(par.grid_power):
                    idx = (t,iP,iL,iA)
                    sol.Cw_priv_remain_couple[idx] = remain_Cw_priv[iP] 
                    sol.Cm_priv_remain_couple[idx] = remain_Cm_priv[iP]
                    sol.C_pub_remain_couple[idx] = remain_C_pub[iP]
                    sol.Vw_remain_couple[idx] = remain_Vw[iP]
                    sol.Vm_remain_couple[idx] = remain_Vm[iP]

    def solve_remain_couple(self,t,M_resources,iL,iP,power,Vw_next,Vm_next,starting_val = None):
        par = self.par

        if t==(par.T-1): # Terminal period
            C_tot = M_resources

        else:
            # objective function
            obj = lambda x: - self.value_of_choice_couple(x[0],t,M_resources,iL,iP,power,Vw_next,Vm_next)[0]
            x0 = np.array([M_resources * 0.8]) if starting_val is None else starting_val

            # optimize
            res = optimize.minimize(obj,x0,bounds=((1.0e-6, M_resources - 1.0e-6),) ,method='SLSQP') 
            C_tot = res.x[0]

        # implied consumption allocation (re-calculation)
        _, Cw_priv, Cm_priv, C_pub, Vw,Vm = self.value_of_choice_couple(C_tot,t,M_resources,iL,iP,power,Vw_next,Vm_next)

        # return objects
        return Cw_priv, Cm_priv, C_pub, Vw, Vm

    def value_of_choice_couple(self,C_tot,t,M_resources,iL,iP,power,Vw_next,Vm_next):
        sol = self.sol
        par = self.par

        love = par.grid_love[iL]
        
        # current utility from consumption allocation
        Cw_priv, Cm_priv, C_pub = intraperiod_allocation(C_tot,iP,sol,par)
        Vw = usr.util(Cw_priv,C_pub,woman,par,love)
        Vm = usr.util(Cm_priv,C_pub,man,par,love)

        # add continuation value
        if t < (par.T-1):
            # savings_vec = np.ones(par.num_shock_love)
            sol.savings_vec[:] = M_resources - C_tot #np.repeat(M_resources - C_tot,par.num_shock_love) np.tile(M_resources - C_tot,(par.num_shock_love,)) 
            love_next_vec = love + par.grid_shock_love

            linear_interp.interp_2d_vec(par.grid_love,par.grid_A , Vw_next, love_next_vec,sol.savings_vec,sol.Vw_plus_vec)
            linear_interp.interp_2d_vec(par.grid_love,par.grid_A , Vm_next, love_next_vec,sol.savings_vec,sol.Vm_plus_vec)

            EVw_plus = sol.Vw_plus_vec @ par.grid_weight_love
            EVm_plus = sol.Vm_plus_vec @ par.grid_weight_love

            Vw += par.beta*EVw_plus
            Vm += par.beta*EVm_plus

        # return
        Val = power*Vw + (1.0-power)*Vm
        return Val , Cw_priv, Cm_priv, C_pub, Vw,Vm
        
    def value_of_choice_single(self,C_tot,M,gender,V_next):
        par = self.par

        # flow-utility
        C_priv = usr.cons_priv_single(C_tot,gender,par)
        C_pub = C_tot - C_priv
        
        Util = usr.util(C_priv,C_pub,gender,par)
        
        # continuation value
        grid_A = par.grid_Aw if gender==woman else par.grid_Am
        A = M - C_tot

        Vnext = linear_interp.interp_1d(grid_A,V_next,A)
        
        # return discounted sum
        return Util + par.beta*Vnext
   
    def simulate(self):
        sol = self.sol
        sim = self.sim
        par = self.par

        for i in range(par.simN):
            for t in range(par.simT):

                # state variables
                if t==0:
                    A_lag = sim.init_A[i]
                    Aw_lag = sim.init_Aw[i]
                    Am_lag = sim.init_Am[i]
                    couple_lag = sim.init_couple[i]
                    power_idx_lag = sim.init_power_idx[i]
                    love = sim.love[i,t] = sim.init_love[i]

                else:
                    A_lag = sim.A[i,t-1]
                    Aw_lag = sim.Aw[i,t-1]
                    Am_lag = sim.Am[i,t-1]
                    couple_lag = sim.couple[i,t-1]
                    power_idx_lag = sim.power_idx[i,t-1]
                    love = sim.love[i,t]
                
                power_lag = par.grid_power[power_idx_lag]

                # first check if they want to remain together and what the bargaining power will be if they do.
                if couple_lag:                   

                    # value of transitioning into singlehood
                    Vw_single = linear_interp.interp_1d(par.grid_Aw,sol.Vw_trans_single[t],Aw_lag)
                    Vm_single = linear_interp.interp_1d(par.grid_Am,sol.Vm_trans_single[t],Am_lag)

                    idx = (t,power_idx_lag)
                    Vw_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                    Vm_couple_i = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                    if ((Vw_couple_i>=Vw_single) & (Vm_couple_i>=Vm_single)):
                        power_idx = power_idx_lag

                    else:
                        # value of partnerhip for all levels of power
                        Vw_couple = np.zeros(par.num_power)
                        Vm_couple = np.zeros(par.num_power)
                        for iP in range(par.num_power):
                            idx = (t,iP)
                            Vw_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vw_remain_couple[idx],love,A_lag)
                            Vm_couple[iP] = linear_interp.interp_2d(par.grid_love,par.grid_A,sol.Vm_remain_couple[idx],love,A_lag)

                        # check participation constraint TODO: should it be the value of REMAINING MARRIED? now it is the value of entering the period as married...
                        Sw = Vw_couple - Vw_single
                        Sm = Vm_couple - Vm_single
                        power_idx = update_bargaining_index(Sw,Sm,power_idx_lag, par)

                    # infer partnership status
                    if power_idx < 0.0: # divorce is coded as -1
                        sim.couple[i,t] = False

                    else:
                        sim.couple[i,t] = True

                else: # remain single

                    sim.couple[i,t] = False

                # update behavior
                if sim.couple[i,t]:
                    
                    # optimal consumption allocation if couple
                    sol_C_tot = sol.C_tot_couple[t,power_idx] 
                    C_tot = linear_interp.interp_2d(par.grid_love,par.grid_A,sol_C_tot,love,A_lag)

                    sim.Cw_priv[i,t], sim.Cm_priv[i,t], C_pub = intraperiod_allocation(C_tot,power_idx,sol,par)
                    sim.Cw_pub[i,t] = C_pub
                    sim.Cm_pub[i,t] = C_pub

                    # update end-of-period states
                    M_resources = usr.resources_couple(A_lag,par) 
                    sim.A[i,t] = M_resources - sim.Cw_priv[i,t] - sim.Cm_priv[i,t] - sim.Cw_pub[i,t]
                    if t<(par.simT-1):
                        sim.love[i,t+1] = love + par.sigma_love*sim.draw_love[i,t+1]

                    # in case of divorce
                    sim.Aw[i,t] = par.div_A_share * sim.A[i,t]
                    sim.Am[i,t] = (1.0-par.div_A_share) * sim.A[i,t]

                    sim.power_idx[i,t] = power_idx
                    sim.power[i,t] = par.grid_power[sim.power_idx[i,t]]

                else: # single

                    # pick relevant solution for single, depending on whether just became single
                    idx_sol_single = t
                    sol_single_w = sol.Cw_tot_trans_single[idx_sol_single]
                    sol_single_m = sol.Cm_tot_trans_single[idx_sol_single]
                    if (power_idx_lag<0):
                        sol_single_w = sol.Cw_tot_single[idx_sol_single]
                        sol_single_m = sol.Cm_tot_single[idx_sol_single]

                    # optimal consumption allocations
                    Cw_tot = linear_interp.interp_1d(par.grid_Aw,sol_single_w,Aw_lag)
                    Cm_tot = linear_interp.interp_1d(par.grid_Am,sol_single_m,Am_lag)
                    
                    sim.Cw_priv[i,t],sim.Cw_pub[i,t] = intraperiod_allocation_single(Cw_tot,woman,par)
                    sim.Cm_priv[i,t],sim.Cm_pub[i,t] = intraperiod_allocation_single(Cm_tot,man,par)

                    # update end-of-period states
                    Mw = usr.resources_single(Aw_lag,woman,par)
                    Mm = usr.resources_single(Am_lag,man,par) 
                    sim.Aw[i,t] = Mw - sim.Cw_priv[i,t] - sim.Cw_pub[i,t]
                    sim.Am[i,t] = Mm - sim.Cm_priv[i,t] - sim.Cm_pub[i,t]

                    # not updated: nans
                    # sim.power[i,t] = np.nan
                    # sim.love[i,t+1] = np.nan 
                    # sim.A[i,t] = np.nan

                    sim.power_idx[i,t] = -1

        # total consumption
        sim.Cw_tot = sim.Cw_priv + sim.Cw_pub
        sim.Cm_tot = sim.Cm_priv + sim.Cm_pub
        sim.C_tot = sim.Cw_priv + sim.Cm_priv + sim.Cw_pub
    

############
# routines #
############
def intraperiod_allocation_single(C_tot,gender,par):
    C_priv = usr.cons_priv_single(C_tot,gender,par)
    C_pub = C_tot - C_priv
    return C_priv,C_pub
 
def intraperiod_allocation(C_tot,iP,sol,par):

    # interpolate pre-computed solution
    j1 = linear_interp.binary_search(0,par.num_Ctot,par.grid_Ctot,C_tot)
    Cw_priv = linear_interp_1d._interp_1d(par.grid_Ctot,sol.pre_Ctot_Cw_priv[iP],C_tot,j1)
    Cm_priv = linear_interp_1d._interp_1d(par.grid_Ctot,sol.pre_Ctot_Cm_priv[iP],C_tot,j1)
    C_pub = C_tot - Cw_priv - Cm_priv 

    return Cw_priv, Cm_priv, C_pub

def solve_intraperiod_couple(C_tot,power,par,starting_val=None):
    
    # setup estimation. Impose constraint that C_tot = Cw+Cm+C
    bounds = optimize.Bounds(0.0, C_tot, keep_feasible=True)
    obj = lambda x: - (power*usr.util(x[0],C_tot-np.sum(x),woman,par) + (1.0-power)*usr.util(x[1],C_tot-np.sum(x),man,par))
    
    # estimate
    x0 = np.array([C_tot/3,C_tot/3]) if starting_val is None else starting_val
    res = optimize.minimize(obj,x0,bounds=bounds)

    # unpack
    Cw_priv = res.x[0]
    Cm_priv = res.x[1]
    C_pub = C_tot - Cw_priv - Cm_priv

    return Cw_priv,Cm_priv,C_pub

def check_participation_constraints(power_idx,power,Sw,Sm,idx_single,idx_couple,list_couple,list_raw,list_single, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        for iP in range(par.num_power):

            # overwrite output for couple
            idx = idx_couple(iP)
            for i,key in enumerate(list_couple):
                list_couple[i][idx] = list_raw[i][iP]

            power_idx[idx] = iP
            power[idx] = par.grid_power[iP]

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        for iP in range(par.num_power):

            # overwrite output for couple
            idx = idx_couple(iP)
            for i,key in enumerate(list_couple):
                list_couple[i][idx] = list_single[i][idx_single]

            power_idx[idx] = -1
            power[idx] = -1.0

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 1 #0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1-1 #par.num_power-1 # in case there is no crossing, this will be the correct value
        for iP in range(par.num_power-1):
            if (Sw[iP]<0) & (Sw[iP+1]>=0):
                Low_w = iP+1
                
            if (Sm[iP]>=0) & (Sm[iP+1]<0):
                Low_m = iP

        # b. interpolate the surplus of each member at indifference points
        # women indifference
        id = Low_w-1
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_w = par.grid_power[id] - Sw[id]/ratio_w
        Sm_at_zero_w = Sm[id] + ratio_m*( power_at_zero_w - par.grid_power[id] )

        # men indifference
        id = Low_m
        denom = (par.grid_power[id+1] - par.grid_power[id])
        ratio_w = (Sw[id+1] - Sw[id])/denom
        ratio_m = (Sm[id+1] - Sm[id])/denom
        power_at_zero_m = par.grid_power[id] - Sm[id]/ratio_m
        Sw_at_zero_m = Sw[id] + ratio_w*( power_at_zero_m - par.grid_power[id] )


        # update the outcomes
        for iP in range(par.num_power):

            # index to store solution for couple 
            idx = idx_couple(iP)
    
            # woman wants to leave
            if iP<Low_w: 
                if Sm_at_zero_w > 0: # man happy to shift some bargaining power

                    for i,key in enumerate(list_couple):
                        if iP==0:
                            list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_w,Low_w-1) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(0)]; # re-use that the interpolated values are identical

                    power_idx[idx] = Low_w
                    power[idx] = power_at_zero_w
                    
                else: # divorce

                    for i,key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single]

                    power_idx[idx] = -1
                    power[idx] = -1.0
                
            # man wants to leave
            elif iP>Low_m: 
                if Sw_at_zero_m > 0: # woman happy to shift some bargaining power
                    
                    for i,key in enumerate(list_couple):
                        if (iP==(Low_m+1)):
                            list_couple[i][idx] = linear_interp_1d._interp_1d(par.grid_power,list_raw[i],power_at_zero_m,Low_m) 
                        else:
                            list_couple[i][idx] = list_couple[i][idx_couple(Low_m+1)]; # re-use that the interpolated values are identical

                    power_idx[idx] = Low_m
                    power[idx] = power_at_zero_m
                    
                else: # divorce

                    for i,key in enumerate(list_couple):
                        list_couple[i][idx] = list_single[i][idx_single]

                    power_idx[idx] = -1
                    power[idx] = -1.0

            else: # no-one wants to leave

                for i,key in enumerate(list_couple):
                    list_couple[i][idx] = list_raw[i][iP]

                power_idx[idx] = iP
                power[idx] = par.grid_power[iP]

def update_bargaining_index(Sw,Sm,iP, par):
    
    # check the participation constraints. Array
    min_Sw = np.min(Sw)
    min_Sm = np.min(Sm)
    max_Sw = np.max(Sw)
    max_Sm = np.max(Sm)

    if (min_Sw >= 0.0) & (min_Sm >= 0.0): # all values are consistent with marriage
        return iP

    elif (max_Sw < 0.0) | (max_Sm < 0.0): # no value is consistent with marriage
        return -1

    else: 
    
        # find lowest (highest) value with positive surplus for women (men)
        Low_w = 0 # in case there is no crossing, this will be the correct value
        Low_m = par.num_power-1 # in case there is no crossing, this will be the correct value
        for _iP in range(par.num_power-1):
            if (Sw[_iP]<0) & (Sw[_iP+1]>=0):
                Low_w = _iP+1
                
            if (Sm[_iP]>=0) & (Sm[_iP+1]<0):
                Low_m = iP

        # update the outcomes
        # woman wants to leave
        if iP<Low_w: 
            if Sm[Low_w] > 0: # man happy to shift some bargaining power
                return Low_w
                
            else: # divorce
                return -1
            
        # man wants to leave
        elif iP>Low_m: 
            if Sw[Low_m] > 0: # woman happy to shift some bargaining power
                return Low_m
                
            else: # divorce
                return -1

        else: # no-one wants to leave
            return iP
