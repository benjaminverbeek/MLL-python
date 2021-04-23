# code to make 3D fit (Actually simplifies to 2D)
# Benjamin Verbeek, 2021-04-19
import math  # NOTE: what do I need? Pi? anything else? take cos, sin from np
import time
import numpy as np
from scipy import optimize      # requires download
from scipy import integrate     # requires download
import numba                    # requires download
from numba import jit           # requires download
from numba.typed import List    # requires download

###### THEORY ######
alpha = 0.753   # assumed.  Had 0.753 before, which stood in formalism_viktor.pdf

@jit(nopython=True) # nopython=True doesn't appear to make a difference but is apparently recommended.
def WSingleTagNum(eta, delta_phi, cos_th, cos_thP):
    '''Normalize this to get the PDF we want to optimize. W is the function from theory'''
    return 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP  # W function
#### END THEORY ####
    
###################
# MC-integrator for normalization factors
@jit(nopython=True)
def MCintegralNum(eta, delta_phi, uniformAngles):
    
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # point is a 2D list
        cos_th, cos_thP = xi
        s += WSingleTagNum(eta, delta_phi, cos_th, cos_thP)
        n += 1
    return 1/n * s * (2)**2    # MC-integral
###################

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
def negLogLikelihood(par, var, pdf, normalizeSeparately=False, normalizationAngles=[]):
    '''Minimize this function for decay parameters to find max of Log-Likelihood for distribution. \n
    par : decay parameters to maximize [list], N-dim \n
    var : dataset of variables (xi) [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every variable is not a separate list, but rather part of a set of variables (i.e. a point)). 
    E.g.:
    >>> var = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> var = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different variables for the pdf. \n
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)'''
    
    t1 = time.time()

    print("--------")
    if normalizeSeparately==True:
        normalization = MCintegralNum(*par, normalizationAngles)    # this one takes a really long time!
        #normalization = integrate.nquad(WSingleTagIntegrable(*par), [[-1,1],[-1,1]])[0] # fast. returns (val, err) so must take [0]
        # NOTE: Can't use non MC because we need to balance for detector?
        print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.2f} seconds.")
    else:
        normalization = 1
    
    @jit(nopython=True)
    def iterativeLL(par, var):
        s = 0  # sum
        eta, delta_phi = par
        for v in var: # iterate over samples of xi
            cos_th, cos_thP = v
            s -= np.log(WSingleTagNum(eta, delta_phi, cos_th, cos_thP)) # log-sum of pdf gives LL. Negative so we minimize.
        return s
    r = iterativeLL(par,var) + len(var)*np.log(normalization) # normalize after
    
    t3 = time.time()
    print(f"One set of parameters done. Took {t3 - t2:.2f} seconds.")    # takes a long time but not AS long as normalization.
    print(f"Total time for one run was {t3 - t1:.2f} seconds.")
    return r

########## MAIN: ########## 
def main():

    start_time = time.time()
    print("Reading input data...")

    ########## READ DATA: ##########
    # Read angle data.
    xi_set = [ List(list(map(float,i.split()))) for i in open("lAngles.txt").readlines() ]    # list (of lists)

    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    print("--- %s seconds ---" % (time.time() - start_time))
    xi_set = List(xi_set)
    # takes approx. 1.3 seconds to read file as list. 3x as long to make it floats

    normalizationAngles = [ List(list(map(float,i.split()))) for i in open("lPHSP_4Pi_testing.txt").readlines() ]    # list (of lists) 
    # NOTE: In numbda List format.
    print("Reading done. Converting for numba.")
    print(normalizationAngles[0])
    normalizationAngles = List(normalizationAngles)   # conversion needed for numba handling. I think unpacking breaks things.
    # conversion and list comprehension w List() takes a LONG time (abt 100 s for 5 million points)

    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print("--- %s seconds ---" % (time.time() - start_time))
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    # Generated data with R=0.91 and delta_phi = 42 deg (0.733 rad)
    # Variables eta, delta-phi
    initial_guess = [0.4, 60*math.pi/180]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7))   # bounds on variables
    q = 2.396 # GeV, reaction energy (momentum transfer)
    mLambda = 1.115683 # GeV, mass of lambda baryon (from PDG-live)
    tau = q**2/(4*mLambda**2)   # form factor #tau = 1.15442015725 # Viktors värde? #tau = 1.1530071615814588 # mitt beräknade 
    tolerance = 10**-6

    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLogLikelihood, initial_guess, (xi_set[0:-1], WSingleTagNum, True, normalizationAngles[0:-1]), tol=tolerance, bounds=bnds)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(res)
    print(f"Initial guess: {initial_guess}")
    print(f"Expected result: {(0.217, 42*math.pi/180)}")
    print("--- Took a total of %s seconds ---" % (time.time() - start_time))
    eta_res = res['x'][0]
    dphi_res = res['x'][1]
    print(f"Result for eta: {eta_res}")
    # TODO: Make it so a message is sent (popup) when program is finished.: import ctypes ? 
    R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)
    print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/math.pi}")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'Variance eta: {hess[0][0]}. Variance delta-phi: {hess[1][1]} (rad)')
    ########## END PRESENT RESULTS ##########

    ########## END MAIN ##########

if __name__ == "__main__":
    main()
