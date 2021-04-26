# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 3D fit (Actually simplifies to 2D)    #
# Benjamin Verbeek, 2021-04-23                          #
# Updated functions to work with numba, now executes    #
# very fast. Slow part is reading input and converting  #
# to numbda lists.                                      #
# NOTE: Multithread input? Could work.                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, comes with Python
import time                     # for timing, comes with Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit           # numba
from numba.typed import List    # numba
##### END IMPORTS #####

alpha = 0.754   # assumed.  Had 0.753 before, which stood in formalism_viktor.pdf
angleDistributionData_filename = "lAngles.txt"  # specify path if not in same folder.
normalizationData_filename = "lPHSP_4Pi.txt"

###### THEORY ######
@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WSingleTagNum(eta, delta_phi, cos_th, cos_thP):
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP  # W function
##### END THEORY #####
    
##### MC INTEGRATOR #####
# MC-integrator for normalization factors
@jit(nopython=True) # numba decorator. Significantly improves performance (~factor 100)
def MCintegralNum(eta, delta_phi, uniformAngles):
    """Monte Carlo integration for normalization, for given parameters and a set of normalization angles."""
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # xi is a 2D list
        cos_th, cos_thP = xi
        s += WSingleTagNum(eta, delta_phi, cos_th, cos_thP) # evaluate W at a bunch of random points and sum.
        n += 1  # count number of points. Could also use len(uniformAngles)
    return 1/n * s * 2*2    # MC-integral: average value of function * area 
                            # (2*2, since cos has range [-1,1]). This area-constant does not affect results.
##### END MC INTEGRATOR #####

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
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)
    Lists (and lists of lists) should be of type typed_list from numba List() to run.'''
    
    t1 = time.time()

    print("--------")
    if normalizeSeparately==True:
        normalization = MCintegralNum(*par, normalizationAngles)
        print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.5f} seconds.")
    else:
        normalization = 1
    
    @jit(nopython=True)
    def iterativeLL(par, var):  # a separate function so numba can optimize it.
        s = 0  # sum
        eta, delta_phi = par
        for v in var: # iterate over samples of xi
            cos_th, cos_thP = v
            s -= np.log(WSingleTagNum(eta, delta_phi, cos_th, cos_thP)) # log-sum of pdf gives LL. Negative so we minimize.
        return s
    r = iterativeLL(par,var) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    
    t3 = time.time()
    print(f"One set of parameters done. Took {t3 - t2:.5f} seconds.")    # takes a long time but not AS long as normalization.
    print(f"Total time for one run was {t3 - t1:.5f} seconds.")
    return r

############# END HELP FUNCTIONS #############

########## MAIN: ########## 
def main():

    start_time = time.time()
    print("Reading input data... \t (this might take a minute)")

    ########## READ DATA: ##########
    # Read angle distribution data. Becomes python list of numba-lists
    xi_set = [ list(map(float,i.split())) for i in open(angleDistributionData_filename).readlines() ]    # list (of numba lists)
    xi_set = np.asarray(xi_set) # much faster
    #for i in open(angleDistributionData_filename).readlines():
    #    xi_set.append(List(map(float,i.split())))   # split line into str's, convert str->float, convert map->List

    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    #print("Converting to numba list...")
    #xi_set = List(xi_set) # conversion needed for numba handling. I think lists break things (converting back and forth).
    print("DONE")
    t2 = time.time()
    print(f"--- {(time.time() - start_time):.3f} seconds ---")

    # Read normalization data
    # NOTE: Almost as fast one-line alternative:       # TODO: How can I speed this up?
    normalizationAngles = [ list(map(float,i.split())) for i in open(normalizationData_filename).readlines() ]    # list (of numba lists) 
    normalizationAngles = np.asarray(normalizationAngles)
    #normalizationAngles = List()
    #for i in open(normalizationData_filename).readlines():
    #    normalizationAngles.append(List(map(float,i.split())))  # SO SLOW!
    #print("Converting to numba list...")
    print(normalizationAngles[0])
    #normalizationAngles = List(normalizationAngles)   # conversion needed for numba handling. I think unpacking breaks things.
    # conversion and list comprehension w List() takes a LONG time (abt 100 s for 5 million points)

    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"--- {(time.time() - start_time):.3f} seconds total for all input data ---")
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    # Generated data with R=0.91 and delta_phi = 42 deg (0.733 rad)
    # Variables eta, delta-phi
    initial_guess = [0.4, 60*PI/180]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7))   # bounds on variables. NOTE: The (-7,7) bound on dPhi is pretty arbitrary.
    q = 2.396 # GeV, reaction energy (momentum transfer)
    mLambda = 1.115683 # GeV, mass of lambda baryon (from PDG-live)
    tau = q**2/(4*mLambda**2)   # form factor #tau = 1.15442015725 # Viktors värde? #tau = 1.1530071615814588 # mitt beräknade 
    tolerance = 10**-6

    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLogLikelihood, initial_guess, (xi_set[0:-1], WSingleTagNum, True, normalizationAngles[0:-1]), tol=tolerance, bounds=bnds)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(res)  # scipy default result structure
    print(f"------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"Initial guess: \t\t {initial_guess}")
    print(f"Expected result: \t {(0.217, 42*PI/180)}") # input to generate data, according to Viktor
    eta_res = res['x'][0]
    dphi_res = res['x'][1]
    print(f"Result for eta: \t {eta_res}")
    R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)   # according to formalism
    print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/PI} deg")
    print("")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'Variance eta: \t\t {hess[0][0]} \nVariance delta-phi: \t {hess[1][1]} (rad)')
    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()
