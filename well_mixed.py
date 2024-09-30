"""
well_mixed.py: file to store definition of the well-mixed model

CONTAINS: - dR_dt: function storing the dynamics of resources
          - dR_dt_maslov: function with reinsertion in environment
          - dR_dt_linear: function storing the dynamics of resources, when linear
          - dR_dt_nomod: function with R dynamics with uptake not regulated by auxotrophies
          - dN_dt: function storing the dynamics of species
          - dN_dt_linear: function storing the dynamics of species, when linear
          - run_wellmixed: function to run the well-mixed simulation

"""

import numpy as np
from scipy import optimize, integrate


#-------------------------------------------------------------------------------------------------------------------
# define chemicals dynamics, monod+uptake aux modulation

def dR_dt(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS dRdt_squared: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()
    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*mu      # modulate uptakes 
            up_eff[i,lim]=mat['uptake'][i,lim] # restore uptake of the limiting one to max

    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((up_eff*R/(1+R)).T,N.T)

    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'],(n_s,1,1))*np.transpose((np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)),axes=(0,2,1))
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    # vector long n_r with produced chemicals
    prod = np.sum(np.einsum('ij,ijk->ik', up_eff*N[:, np.newaxis]*R/(1+R)*param['w']*param['l'], D_s_norma),axis=0)/param['w'] 
    
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    # sum
    dRdt_squared=(ext+prod-out)**2
    dRdt_squared[np.abs(dRdt_squared)<1e-14]=0

    return ext+prod-out

#-------------------------------------------------------------------------------------------------------------------
# define chemicals dynamics, monod+uptake aux modulation

def dR_dt_maslov(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS dRdt_squared: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    prod = np.zeros((n_r))
    
    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'],(n_s,1,1))*np.transpose((np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)),axes=(0,2,1))
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            l_eff=param['l'].copy()*mu + (1-mu)     # modulate uptakes 
            l_eff[lim]=param['l'].copy()[lim]
            l_eff[lim]=param['l'][lim].copy()     # restore uptake of the limiting one to max
            prod += np.dot(N[i]*mat['uptake'][i]*R/(1+R)*param['w']*l_eff,(D_s_norma[i].T))*1/param['w']
        else:
            prod += np.dot(N[i]*mat['uptake'][i]*R/(1+R)*param['w']*param['l'],(D_s_norma[i].T))*1/param['w']

    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((mat['uptake']*R/(1+R)).T,N.T)
    out[np.abs(out)<1e-14]=0
        
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    # sum
    dRdt_squared=(ext+prod-out)**2
    dRdt_squared[np.abs(dRdt_squared)<1e-14]=0

    return ext+prod-out

#-------------------------------------------------------------------------------------------------------------------
# define chemicals dynamics, monod+uptake aux modulation only partial: a part of uptake is independent on the 
# amino acid concentration, the rest is modulated

def dR_dt_partial(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS dRdt_squared: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()
    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*(param['alpha']+(1-param['alpha'])*mu)      # modulate uptakes only partially
            up_eff[i,lim]=mat['uptake'][i,lim]                                     # restore uptake of the limiting one to max


    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((up_eff*R/(1+R)).T,N.T)

    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'],(n_s,1,1))*np.transpose((np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)),axes=(0,2,1))
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    prod = np.zeros((n_r))
    # vector long n_r with produced chemicals
    for i in range(n_s):
        prod += np.dot(up_eff[i]*N[i]*R/(1+R)*param['w']*param['l'], D_s_norma[i].T)/param['w']
    
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    # sum
    dRdt_squared=(ext+prod-out)**2
    dRdt_squared[np.abs(dRdt_squared)<1e-14]=0

    return ext+prod-out


#-------------------------------------------------------------------------------------------------------------------
# define chemicals dynamics, without auxotrophies modulation on uptake

def dR_dt_nomod(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS dRdt_squared: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()

    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((up_eff*R/(1+R)).T,N.T)
    out[np.abs(out)<1e-14]=0

    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'],(n_s,1,1))*np.transpose((np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)),axes=(0,2,1))
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    # vector long n_r with produced chemicals
    prod = np.zeros((n_r))
    # vector long n_r with produced chemicals
    for i in range(n_s):
        prod += np.dot(up_eff[i]*N[i]*R/(1+R)*param['w']*param['l'], D_s_norma[i].T)/param['w']
    
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    # sum
    dRdt_squared=(ext+prod-out)**2
    dRdt_squared[np.abs(dRdt_squared)<1e-14]=0

    return ext+prod-out

#-------------------------------------------------------------------------------------------------------------------
# define chemicals dynamics-linear (modulation still holds non linear)

def dR_dt_linear(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS dRdt_squared: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()
    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*mu      # modulate uptakes 
            up_eff[i,lim]=mat['uptake'][i,lim] # restore uptake of the limiting one to max

    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((up_eff*R).T,N.T)

    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'].T,(n_s,1,1))*(np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)) 
    D_s_norma = np.zeros((n_s,n_r,n_r))
    for i in range(n_s):
        sums = np.sum(D_species[i], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_s_norma[i] = np.where(sums != 0, D_species[i] / sums, D_species[i])

    # vector long n_r with produced chemicals
    prod = np.sum(np.einsum('ij,ijk->ik', up_eff*R*param['w']*param['l'], D_s_norma),axis=0)/param['w'] 
    
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    # sum
    dRdt_squared=(ext+prod-out)**2
    dRdt_squared[np.abs(dRdt_squared)<1e-14]=0

    return ext+prod-out

#--------------------------------------------------------------------------------------------------
# define species dynamics

def dN_dt(t,N,R,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS N*(growth_vector-1/param['tau_s']), vector, n_s, the new state of species, n_s

    """
    
    n_s = N.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()
    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*mu      # modulate uptakes 
            up_eff[i,lim]=mat['uptake'][i,lim] # restore uptake of the limiting one to max

    # effect of resources
    growth_vector = param['g']*(np.sum(param['w']*(1-param['l'])*up_eff*mat['sign']*R/(1+R),axis=1)) 

    # sum (remember to take dilution away if no species chemostat)
    dNdt = N*(growth_vector-param['m'])
    dNdt[np.abs(dNdt)<1e-10]=0

    return dNdt

#--------------------------------------------------------------------------------------------------
# define species dynamics when linear 

def dN_dt_linear(t,N,R,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS N*(growth_vector-1/param['tau_s']), vector, n_s, the new state of species, n_s

    """
    
    n_s = N.shape[0]

    # check essential nutrients presence (at each site)
    up_eff = mat['uptake'].copy()
    for i in range(n_s):
        # calculate essential nutrients modulation for each species (context-dependent uptake)
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*mu      # modulate uptakes 
            up_eff[i,lim]=mat['uptake'][i,lim] # restore uptake of the limiting one to max

    # effect of resources
    growth_vector = param['g']*(np.sum(param['w']*(1-param['l'])*up_eff*mat['sign']*R,axis=1)-param['m']) 

    # sum
    dNdt = N*(growth_vector)
    dNdt[np.abs(dNdt)<1e-14]=0

    return dNdt


#--------------------------------------------------------------------------------------------------
# define species dynamics when linear 

def dN_dt_maslov(t,N,R,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS N*(growth_vector-1/param['tau_s']), vector, n_s, the new state of species, n_s

    """
    
    n_s = N.shape[0]
    n_r = R.shape[0]

    l = np.zeros((n_s,n_r))

    # check essential nutrients presence (at each site)
    for i in range(n_s):

        l[i] = param['l'].copy()
        
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R[mat['ess'][i]==1]/(R[mat['ess'][i]==1]+1))]
            l_eff=param['l'].copy()*mu + (1-mu)     # modulate uptakes 
            l_eff[lim]=param['l'][lim].copy()     # restore uptake of the limiting one to max
            l[i] = l_eff
                

    # effect of resources
    growth_vector = param['g']*(np.sum(param['w']*(1-l)*mat['uptake']*mat['sign']*R/(1+R),axis=1))-param['m']

    # sum
    dNdt = N*(growth_vector)
    dNdt[np.abs(dNdt)<1e-14]=0

    return dNdt

#-----------------------------------------------------------------------------------------------
# function for the whole simulation

def run_wellmixed(N0,param,mat,dR,dN,maxiter):

    """
    N0: initial state vector of species n_s
    param,mat: matrices and parameters dictionaries
    dR: function, resources dynamics
    dN: function, species dynamics
    maxiter: int, maximum number of iterations accepted

    RETURNS N: list, contains vectors n_s of species time series
            R: list, contains vectors n_r of chemnicals time series

    """

    guess = param['guess_wm']
    N = [N0.copy()]
    R = [guess]

    N_prev = N0
    frac_prev = N0/np.sum(N0)

    i = 0
    j = 0
 
    while(True):

        print("N_iter_wm %d \r" % (i), end='')

        drdt = dR(guess,N_prev,param,mat)
        # solve eq. R
        #if (np.abs(drdt)<1e-14).all():
            #R_eq = R[-1]
        #else:
        R_eq = optimize.least_squares(dR, guess, args=(N_prev,param,mat),bounds = (0,np.inf)).x
        R_eq[np.abs(R_eq)<1e-14]=0

        # integrate N one step
        dndt = dN(0, N_prev, np.array(R_eq), param, mat)
        if ((np.abs(dndt)<1e-14).all() and i>2):
            break
        N_out = integrate.solve_ivp(dN, (0,0.1), N_prev, method='RK23', args=(np.array(R_eq),param,mat))
        N_out = N_out.y[:, -1]
        N_out[N_out<1e-14]=0

        # stop simulation if real abundances converge (after 1000 steps)
        if (np.abs(N_prev-N_out)<1e-6).all() and i>1000:
            j +=1
        if j>5000:
            break

        N_prev = N_out
        guess = R_eq
  
        N.append(N_out)
        R.append(R_eq)

        i +=1

        # stop simulation when fractional abundances converge
        #frac_new = N_out/np.sum(N_out)
        #if (np.abs(frac_new-frac_prev)<1e-8).all() and i >1000:
        #    j += 1
        #if j>10000:
        #    break
        #frac_prev = frac_new.copy()

        if i>maxiter:
            break

    N, R = np.array(N),np.array(R)

    return N,R







