# code to make 3D fit (Actually simplifies to 2D)
# Benjamin Verbeek, 2021-04-19
import math
import os
import time
import numpy as np
from scipy import optimize
#from scipy.sparse import csc_matrix as csc

###### THEORY #####
alpha = 0.753   # assumed.

F0 = lambda cos_th, cos_thP : 1
F1 = lambda cos_th, cos_thP : math.sin(math.acos(cos_th)) * cos_th * cos_thP
F2 = lambda cos_th, cos_thP : (cos_th)**2

def WSingleTag(eta, delta_phi, xi):
    return F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  # W function
###################

###################
# MC-integrator for normalization factors
def MCintegral(par, uniformAngles):
    s = 0   # sum
    n = 0   # number of points
    for point in uniformAngles: # point is a 2D list
        s += WSingleTag(*par, point)
        n += 1
    return 1/n * s * (2*math.pi)**2    # MC-integral

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
    
    print("--------")
    if normalizeSeparately==True:
        normalization = MCintegral(par, normalizationAngles)    # this one takes a really long time!
        print("One normalization done...")
    else:
        normalization = 1

    s = 0  # sum
    for v in var: # iterate over samples of xi
        # * unpacks the list of arguments
        s -= np.log(pdf(*par, v)/normalization) # log-sum of pdf gives LL. Negative so we minimize.
    print("One set of parameters done.")    # takes a long time but not AS long as normalization.
    return s

# MAIN
def main():

    start_time = time.time()
    print("Reading input data...")

    # Read angle data.
    xi_set = [ list(map(float,i.split())) for i in open("lAngles.txt").readlines() ]    # list (of lists)

    print("Finished reading.")
    print(xi_set[0])
    print(f"Number of measurement points: {len(xi_set)}")
    print("--- %s seconds ---" % (time.time() - start_time))
    # takes approx. 1.3 seconds to read file as list. 3x as long to make it floats
    normalizationAngles = [ list(map(float,i.split())) for i in open("lPHSP_4Pi.txt").readlines() ]    # list (of lists)
    # NOTE: cannot both be generators as it would create two open files? Or can they? Furthermore, going thru them
    # multiple times ruins it...
    print(normalizationAngles[0])
    print(f"Number of random angles for normalization: {len(normalizationAngles)}")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Optimize:
    # Generated data with R=0.91 and delta_phi = 42 deg (0.733 rad)
    # '''
    # Variables eta, delta-phi
    initial_guess = [0.4, 60*math.pi/180]
    print(f"Initial guess: {initial_guess}")
    bnds = ((-1,1),(-7,7))   # bounds on variables
    # eta = (tau - R^2)/(tau + R^2), 
    # where tau = q^2/(4m^2), q = 2.396 GeV here (momentum transfer), m = 1.115683 GeV (1115.683 MeV)
    # ===> eta = 0.217 (and delta_phi = 0.733 rad) is to be expected. 

    print("Optimizing...")
    res = optimize.minimize(negLogLikelihood, initial_guess, (xi_set[0:], WSingleTag, True, normalizationAngles[0:]), tol=10**-4, bounds=bnds)
    # '''
    print(res)
    print(f"Initial guess: {initial_guess}")
    print(f"Expected result: {(0.217, 42*math.pi/180)}")
    print("--- Took a total of %s seconds ---" % (time.time() - start_time))
    eta_res = res['x'][0]
    dphi_res = res['x'][1]
    print(f"Result for eta: {eta_res}")
    tau = 1.28717847534
    R = tau**(0.5) * ((1-eta_res)/(1+eta_res))**(0.5)
    print(f"Yielding R = {R}")
    print(f"delta-phi = {dphi_res} rad, or delta-phi = {dphi_res*180/math.pi}")
    hess = (res['hess_inv']).todense()
    print(hess)
    print(f'Variance eta: {hess[0][0]}. Variance delta-phi: {hess[1][1]} (rad)')

if __name__ == "__main__":
    main()
