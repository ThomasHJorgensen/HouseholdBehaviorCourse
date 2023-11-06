# set gender indication as globals
woman = 1
man = 2

############################
# User-specified functions #
############################
def util(c_priv,c_pub,gender,par,love=0.0):
    if gender == woman:
        rho = par.rho_w
        phi = par.phi_w
        alpha1 = par.alpha1_w
        alpha2 = par.alpha2_w
    else:
        rho = par.rho_m
        phi = par.phi_m
        alpha1 = par.alpha1_m
        alpha2 = par.alpha2_m
    
    return ((alpha1*c_priv**phi + alpha2*c_pub**phi)**(1.0-rho))/(1.0-rho) + love

def resources_couple(A,par):
    return par.R*A + par.inc_w + par.inc_m

def resources_single(A,gender,par):
    income = par.inc_w
    if gender == man:
        income = par.inc_m

    return par.R*A + income

def cons_priv_single(C_tot,gender,par):
    # closed form solution for intra-period problem of single.
    if gender == woman:
        rho = par.rho_w
        phi = par.phi_w
        alpha1 = par.alpha1_w
        alpha2 = par.alpha2_w
    else:
        rho = par.rho_m
        phi = par.phi_m
        alpha1 = par.alpha1_m
        alpha2 = par.alpha2_m   
    
    return C_tot/(1.0 + (alpha2/alpha1)**(1.0/(1.0-phi)) )
