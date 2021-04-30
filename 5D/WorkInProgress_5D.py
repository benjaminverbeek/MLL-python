# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to make 5D fit (Double-tag BESIII)            #
# Benjamin Verbeek, 2021-04-30                          #
# Work in progress version. Messy, functional, got      #
# everython it needs. Used for reference.               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

##### IMPORTS #####
# Imports necessary modules
from math import pi as PI       # for pi, comes with Python
import time                     # for timing, comes with Python
import numpy as np              # requires download, e.g. "$ pip3 install numpy". For scipy and efficient funcs.
from numpy import sin, cos, arccos
from scipy import optimize      # requires download, e.g. "$ pip3 install scipy". For optimization of LL.
import numba                    # requires download, e.g. "$ pip3 install numba". For efficient execution.
from numba import jit           # numba
##### END IMPORTS #####

# Set some parameters for the fit.
angleDistributionData_filename = "mcsig100k_JPsi_LLbar.dat"  # specify path if not in same folder.
normalizationData_filename = "mcphsp1000k_JPsi_LLbar.dat"
numberedInput = True        # Is one column of the input data just numbering? Specify that here.

numberedInput = int(numberedInput)  # True - 1, False - 0. This is how many colums to skip in indata files. Can be specified manually further down.

###### THEORY ######
@jit(nopython=True)
def F0(th,th1,th2,ph1,ph2): return 1

@jit(nopython=True)
def F1(th,th1,th2,ph1,ph2): return ( (sin(th))**2 * sin(th1) * sin(th2) * \
            cos(ph1) * cos(ph2) + (cos(th))**2 * cos(th1) * cos(th2) )

@jit(nopython=True)
def F2(th,th1,th2,ph1,ph2): return sin(th) * cos(th) * (sin(th1) * cos(th2) * \
            cos(ph1) + cos(th1) * sin(th2) * cos(ph2))

@jit(nopython=True)
def F3(th,th1,th2,ph1,ph2): return sin(th) * cos(th) * sin(th1) * sin(ph1)

@jit(nopython=True)
def F4(th,th1,th2,ph1,ph2): return sin(th) * cos(th) * sin(th2) * sin(ph2)

@jit(nopython=True)
def F5(th,th1,th2,ph1,ph2): return (cos(th))**2

@jit(nopython=True)
def F6(th,th1,th2,ph1,ph2): return cos(th1) * cos(th2) - (sin(th))**2 * \
            sin(th1) * sin(th2) * sin(ph1) * sin(ph2)

@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2):
    xi = (th,th1,th2,ph1,ph2)
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return F0(*xi) + alpha*F5(*xi) \
        + alpha1*alpha2 * (F1(*xi) + ((1-alpha**2)**0.5) * cos(dPhi) * F2(*xi) + alpha*F6(*xi)) \
        + ((1-alpha**2)**0.5) * sin(dPhi) * (alpha1*F3(*xi) + alpha2*F4(*xi))    # W function

@jit(nopython=True) # Applies numba magic. nopython=True doesn't appear to make a difference but is apparently recommended.
def WDoubleTag2(alpha,dPhi,alpha1,alpha2 , th,th1,th2,ph1,ph2):
    xi = (th,th1,th2,ph1,ph2)
    '''Normalize this to get the PDF to optimize. W is the function from theory (Fäldt, Kupsc)'''
    # https://arxiv.org/pdf/1702.07288.pdf
    return alpha1*alpha2 * ((sin(th))**2 * sin(th1)*sin(th2)*cos(ph1)*cos(ph2) + (cos(th))**2 * cos(th1)*cos(th2) + (1-alpha**2)**0.5 * cos(dPhi) * sin(th) * cos(th) * (sin(th1)*cos(th2)*cos(ph1) + cos(th1)*sin(th2)*cos(ph2)) + alpha*(cos(th1)*cos(th2) - (sin(th))**2 * sin(th1)*sin(th2)*sin(ph1)*sin(ph2))) + alpha1 * (1-alpha**2)**0.5 * sin(dPhi)*sin(th)*cos(th)*sin(th1)*sin(ph1) + alpha2 * (1-alpha**2)**0.5 * sin(dPhi) * sin(th)*cos(th)*sin(th2)*sin(ph2) + alpha*(cos(th))**2 + 1                                                                                          


######### ALT THEORY: ##########
# here alpha = eta = alpha_psi
@jit(nopython=True)
def C00(alpha, dPhi, th): return  2*(1 + alpha * cos(th)**2)

@jit(nopython=True)
def C02(alpha, dPhi, th): return  2*(1-alpha**2)**0.5 * sin(th)*cos(th)*sin(dPhi)

@jit(nopython=True)
def C11(alpha, dPhi, th): return  2*sin(th)**2

@jit(nopython=True)
def C13(alpha, dPhi, th): return  2*(1-alpha**2)**0.5 * sin(th)*cos(th)*cos(dPhi)

@jit(nopython=True)
def C20(alpha, dPhi, th): return  -1*C02(alpha,dPhi,th)

@jit(nopython=True)
def C22(alpha, dPhi, th): return  alpha*C11(alpha,dPhi,th)

@jit(nopython=True)
def C31(alpha, dPhi, th): return  -1*C13(alpha,dPhi,th)

@jit(nopython=True)
def C33(alpha, dPhi, th): return -2*(alpha + cos(th)**2)
##########
# a-funcs should have alpha1/2, th1/2, ph1/2 depending on lambda or lambda-bar
@jit(nopython=True)
def a00(alpha, th, ph): return 1

@jit(nopython=True)
def a10(alpha, th, ph): return alpha * cos(ph) * sin(th)

@jit(nopython=True)
def a20(alpha, th, ph): return alpha * sin(th) * sin(ph)

@jit(nopython=True)
def a30(alpha, th, ph): return alpha * cos(th)
##########
@jit(nopython=True)
def altWDoubleTag(alpha, dPhi, alpha1, alpha2, th, th1, ph1, th2, ph2):
    return C00(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C02(alpha, dPhi, th)/2 * a00(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C11(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C13(alpha, dPhi, th)/2 * a10(alpha1, th1, ph1) * a30(alpha2, th2, ph2) + \
    C20(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a00(alpha2, th2, ph2) + \
    C22(alpha, dPhi, th)/2 * a20(alpha1, th1, ph1) * a20(alpha2, th2, ph2) + \
    C31(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a10(alpha2, th2, ph2) + \
    C33(alpha, dPhi, th)/2 * a30(alpha1, th1, ph1) * a30(alpha2, th2, ph2)

##### END THEORY #####
''' # still dont get same values...
pars = [0.46, 0.785398, 0.75, -0.75]
#pars = [0,0,0,0]
#angs = [0,0,0,0,0]
#angs = [0.0321418,        2.31065,        2.80985,        2.02057,       -1.68605]
angs = [ 1.94843,       0.916242,       -2.08242,        1.53144,      -0.153751]
r1 = altWDoubleTag(*pars, *angs)
r2 = WDoubleTag2(*pars, *angs)
print(f"W-func evaluated at {pars}, {angs}, yielding:")
print(f"v1:    {r1}      v2:     {r2}      equal: {r1==r2}")
print("abort.")
'''

##### MC INTEGRATOR #####
# MC-integrator for normalization factors
@jit(nopython=True) # numba decorator. Significantly improves performance (~factor 100)
def MCintegral(alpha,dPhi,alpha1,alpha2, uniformAngles):
    """Monte Carlo integration for normalization, for given parameters and a set of normalization angles."""
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # xi is a 5D list
        th,th1,ph1,th2,ph2 = xi
        s += altWDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,ph1,th2,ph2) # evaluate W at a bunch of random points and sum.
        n += 1  # count number of points. Could also use len(uniformAngles)
    return 1/n * s #* (2*PI)**5    # MC-integral: average value of function * area # NOTE: area is wrong for 5D but results should be same? Is corrected
                            # (2**5, since cos has range [-1,1]). This area-constant does not affect results.
                            # NOTE: this area was omitted in ROOT. It doesn't affect results since it just adds onto all terms.
##### END MC INTEGRATOR #####

@jit(nopython=True)
def iterativeLL(par, var):  # a separate function so numba can optimize it.
    s = 0  # sum
    alpha,dPhi,alpha1,alpha2 = par
    for v in var: # iterate over samples of xi
        th,th1,ph1,th2,ph2 = v
        s -= np.log(altWDoubleTag(alpha,dPhi,alpha1,alpha2 , th,th1,ph1,th2,ph2)) # log-sum of pdf gives LL. Negative so we minimize.
    return s

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
def negLL(par, var, pdf, normalizeSeparately=False, normalizationAngles=[]):
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
        normalization = MCintegral(*par, normalizationAngles)
        #print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.5f} seconds. Norm = {normalization}")
    else:
        normalization = 1
    
    r = iterativeLL(par,var) + len(var)*np.log(normalization) # normalize after; -log(W_i/norm) = -log(W_i) + log(norm) 
    t3 = time.time()
    print(f"LL: \t {r}")
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
    xi_set = [ list(map(float,i.split()))[numberedInput:] for i in open(angleDistributionData_filename).readlines() ]    # list (of lists)
    # Iterate thru lines of datafile, for each line, split it into list of number contents,
    # map the content of that list from str -> float, convert map object -> list, 
    # skip first if it is numbered input, all in list comprehension.
    xi_set = np.asarray(xi_set) # converts to numpy.array. Much faster than numba typed list.

    print("Finished reading.")
    print(xi_set[0:5])
    print(f"Number of measurement points: {len(xi_set)}")
    print("DONE")
    t2 = time.time()
    print(f"--- {(t2 - start_time):.3f} seconds ---")

    # Read normalization data
    normalizationAngles = [ list(map(float,i.split()))[numberedInput:] for i in open(normalizationData_filename).readlines() ]    # list (of lists) 
    normalizationAngles = np.asarray(normalizationAngles) # needed for numba. Fix datatype.

    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print(f"--- {(time.time() - t2):.3f} seconds for normalization data ---")
    print(f"--- {(time.time() - start_time):.3f} seconds total for all input data ---")
    ########## END READ DATA ##########

    ########## OPTIMIZE: ##########
    #  input parametervärden 1, 2, 3, 4 = 0.460, 0.740, 0.754, -0.754
    # Variables alpha, dPhi, alpha1, alpha2

    initial_guess = [0.3, PI/8, 0.55, -0.55]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-PI,PI),(-1,1),(-1,1))   # bounds on variables. NOTE: Taken same as patrik
    #ops = {'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}
    tolerance = 10**-6
    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLL, initial_guess, (xi_set[0:], altWDoubleTag, True, normalizationAngles[0:]), tol=tolerance, bounds=bnds)#, method='L-BFGS-B')#, options=ops)
    ########## END OPTIMIZE ##########

    ########## PRESENT RESULTS: ##########
    print(res)  # scipy default result structure
    print(f"------ TOOK A TOTAL OF {time.time() - start_time:.3f} SECONDS ------")
    print(f"Initial guess: \t\t {initial_guess}")
    print(f"Expected result: \t {(0.460, PI/4, 0.75, -0.75)}") # input to generate data, according to Patrik
    alpha_res = res['x'][0]
    dphi_res = res['x'][1]
    alpha1_res = res['x'][2]
    alpha2_res = res['x'][3]
    print(f"Result for alpha: \t {alpha_res}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/PI} deg")
    print(f"Result for alpha1: \t {alpha1_res}")
    print(f"Result for alpha2: \t {alpha2_res}")
    
    print("")
    hess = (res['hess_inv']).todense()
    print("Inverse Hessian:")
    print(hess)
    print(f'Variance alpha: \t {hess[0][0]} \nVariance delta-phi: \t {hess[1][1]} (rad)')
    ########## END PRESENT RESULTS ##########

########## END MAIN ##########

if __name__ == "__main__":  # doesn't run if imported.
    main()