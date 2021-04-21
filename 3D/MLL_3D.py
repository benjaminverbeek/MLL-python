# code to make 3D fit (Actually simplifies to 2D)
# Benjamin Verbeek, 2021-04-19
import math
import os
import time
import numpy as np
from scipy import optimize
from scipy import integrate
from matplotlib import pyplot as plt
#from scipy.sparse import csc_matrix as csc

###### THEORY ######
alpha = 0.753   # assumed.

F0 = lambda cos_th, cos_thP : 1
F1 = lambda cos_th, cos_thP : math.sin(math.acos(cos_th)) * cos_th * cos_thP
F2 = lambda cos_th, cos_thP : (cos_th)**2

def WSingleTag(eta, delta_phi, xi):
    '''Normalize this to get the PDF we want to optimize. Xi is the set of angles, packed as a list/tuple/...'''
    return F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  # W function

def WSingleTagIntegrable(eta, delta_phi):
    '''Returns a function to integrate with defined parameters. Requiered format for scipy.integrate. \n
    Returned function depends only on xi.'''

    # xi is packed and then unpacked.
    return (lambda *xi : F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  )

def WSingleTagIntegrableMap(eta, delta_phi):
    '''Returns a function to integrate with defined parameters. Requiered format for scipy.integrate. \n
    Returned function depends only on xi.'''

    # "xi is packed and then unpacked."   # CHANGED: only change to above is not packing first.
    return (lambda xi : F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  )

#### END THEORY ####
    
###################
# MC-integrator for normalization factors
def MCintegral(par, uniformAngles):
    s = 0   # sum
    n = 0   # number of points
    for point in uniformAngles: # point is a 2D list
        s += WSingleTag(*par, point)
        n += 1
    return 1/n * s * (2)**2    # MC-integral    # NOTE: the constant (2*math.pi)**2 doesn't matter for max? # PI^2 was off!

def MCintegralImproved(par,uniformAngles):
    func = WSingleTagIntegrableMap(par)
    uAng = uniformAngles.copy()
    res = list(map(func, uAng))
    return 4*sum(res)/len(uniformAngles)    # area 2x2

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
        #normalization = MCintegral(par, normalizationAngles)    # this one takes a really long time!
        normalization = integrate.nquad(WSingleTagIntegrable(*par), [[-1,1],[-1,1]])[0] # fast. returns (val, err) so must take [0]
        # NOTE: Can't use non MC because we need to balance for detector?
        print(normalization)
        t2 = time.time()
        print(f"One normalization done... took {t2 - t1:.2f} seconds.")
    else:
        normalization = 1
    '''
    s = 0  # sum
    for v in var: # iterate over samples of xi
        # * unpacks the list of arguments
        s -= np.log(pdf(*par, v)/normalization) # log-sum of pdf gives LL. Negative so we minimize.
    t3 = time.time()    # TODO: vectorized function calls?
    '''
    # attempt to use map instead of iterating...
    print("ok0")
    func = WSingleTagIntegrableMap(*par)
    tempVar = var.copy()
    print("ok1")
    tempVar = list(map(func, tempVar))    # applies W       # takes time
    print("ok2")
    tempVar = list(map(np.log, tempVar))  # applies log     # takes time
    print("ok3")
    s = -1*sum(tempVar) + len(var)*np.log(normalization)   # sum, minus, normalize.
    print("ok4")
    t3 = time.time()
    print(f"One set of parameters done. Took {t3 - t2:.2f} seconds.")    # takes a long time but not AS long as normalization.
    print(f"Total time for one run was {t3 - t1:.2f} seconds.")
    return s

# MAIN
def main():

    start_time = time.time()
    print("Reading input data...")

    # Read angle data.
    xi_set = [ list(map(float,i.split())) for i in open("lAngles.txt").readlines() ]    # list (of lists)
    #xi_set = np.asarray(xi_set) # TODO: optimize.  Took a total of 835.8386192321777 seconds  without np.array (1mil norm.)
    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    print("--- %s seconds ---" % (time.time() - start_time))
    # takes approx. 1.3 seconds to read file as list. 3x as long to make it floats
    normalizationAngles = [ list(map(float,i.split())) for i in open("lPHSP_4Pi.txt").readlines() ]    # list (of lists)
    #normalizationAngles = np.asarray(normalizationAngles)
    # NOTE: cannot both be generators as it would create two open files? Or can they? Furthermore, going thru them
    # multiple times ruins it...
    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    # NOTE: The normalization angles are not angles but rather cos(angles).
    print("--- %s seconds ---" % (time.time() - start_time))

    # Optimize:
    # Generated data with R=0.91 and delta_phi = 42 deg (0.733 rad)
    # '''
    # Variables eta, delta-phi
    initial_guess = [0.4, 60*math.pi/180]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7))   # bounds on variables
    q = 2.396 # GeV, reaction energy (momentum transfer)
    mLambda = 1.115683 # GeV, mass of lambda baryon (from PDG-live)
    tau = q**2/(4*mLambda**2)   # form factor
    tolerance = 10**-6

    # int2res = integrate.nquad(WSingleTagIntegrable(0.2, 42*math.pi/180), [[-1,1],[-1,1]])   # NOTE: WORKS
    #print(f"Scipy integral: {int2res}")
    #print(f"MC integral: {MCintegral([0.2, 42*math.pi/180], normalizationAngles)}")
    # eta = (tau - R^2)/(tau + R^2), 
    # where tau = q^2/(4m^2), q = 2.396 GeV here (momentum transfer), m = 1.115683 GeV (1115.683 MeV)
    # ===> eta = 0.217 (and delta_phi = 0.733 rad) is to be expected. 

    print("Optimizing...")
    # scipy existing minimizing function. 
    res = optimize.minimize(negLogLikelihood, initial_guess, (xi_set[0:], WSingleTag, True, normalizationAngles[0:1]), tol=tolerance, bounds=bnds)
    # '''
    # PRESENT RESULTS:
    print(res)
    print(f"Initial guess: {initial_guess}")
    print(f"Expected result: {(0.217, 42*math.pi/180)}")
    print("--- Took a total of %s seconds ---" % (time.time() - start_time))
    eta_res = res['x'][0]
    dphi_res = res['x'][1]
    print(f"Result for eta: {eta_res}")
    #tau = 1.28717847534    # todo: Check where this value went wrong! # NOTE: Found. Missed square. TODO: Add the calculation in program.
    # TODO: Rerun and check functionality
    # TODO: Make it so a message is sent (popup) when program is finished.: import ctypes ? 
    #tau = 1.15442015725 # Viktors värde?
    #tau = 1.1530071615814588 # mitt beräknade 
    R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)
    print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/math.pi}")
    hess = (res['hess_inv']).todense()
    print(hess)
    print(f'Variance eta: {hess[0][0]}. Variance delta-phi: {hess[1][1]} (rad)')

if __name__ == "__main__":
    main()
