# code to make 3D fit
# Benjamin Verbeek
import math
import os
import time
import numpy as np
from scipy import optimize

alpha = 0.753   # assumed.

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
def negLogLikelihood(par, var, pdf):
    '''Minimize this function for decay parameters to find max of Log-Likelihood for distribution. \n
    par : decay parameters to maximize [list], N-dim \n
    var : dataset of variables (xi) [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every variable is not a separate list, but rather part of a set of variables (i.e. a point)). 
    E.g.:
    >>> var = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> var = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different variables for the pdf. \n
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)'''

    s = 0  # sum
    for v in var: # iterate over samples of xi
        # * unpacks the list of arguments
        s -= np.log(pdf(*par, *v)) # log-sum of pdf gives LL. Negative so we minimize.
    return s

F0 = lambda cos_th, cos_thP : 1
F1 = lambda cos_th, cos_thP : math.sin(math.acos(cos_th)) * cos_th * cos_thP
F2 = lambda cos_th, cos_thP : (cos_th)**2


def WSingleTag(alpha, eta, delta_phi, xi):
    return F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  # W function
'''
def WSingleTag(alpha, eta, delta_phi, xi):
    F0 = lambda cos_th, cos_thP : 1
    F1 = lambda cos_th, cos_thP : math.sin(math.acos(cos_th)) * cos_th * cos_thP
    F2 = lambda cos_th, cos_thP : (cos_th)**2
    return F0(*xi) + eta * F2(*xi) + alpha * (1 - eta**2)**(0.5) * math.sin(delta_phi) * F1(*xi)  # W function
'''

def main():

    start_time = time.time()
    print("Reading input data...")
    #print(os.getcwd())

    # Read angle data.
    xi_set = [ list(map(float,i.split())) for i in open("lAngles.txt").readlines() ]    # list (of lists)

    print("Finished reading.")
    print(xi_set[0])
    print(len(xi_set))
    print("--- %s seconds ---" % (time.time() - start_time))
    # takes approx. 1.3 seconds to read file as list. 3x as long to make it floats
    #print(os.getcwd())
    normalizationAngles = [ list(map(float,i.split())) for i in open("lPHSP_4Pi.txt").readlines() ]    # list (of lists)
    # NOTE: cannot both be generators as it would create two open files? Or can they?
    print(normalizationAngles[0])
    print(len(normalizationAngles))
    print("--- %s seconds ---" % (time.time() - start_time))
    # TODO: Get the PDF?? Initial guess is what exactly? PDF depends only on R and delta_phi, 
    # not alpha and eta?
    # ... ---> how do I normalize W? I got some data for that ("flat angular distribuion") 
    # ... ---> do I then just divide W by the sum of these 

    # Optimize:
    # Generated data with R=0.91 and delta_phi = 42 deg (0.66 rad)
    # '''
    # Variables alpha, eta, delta-phi
    initial_guess = [0.6, 0.2, 42*math.pi/180]
    bnds = ((-1,1),(-1,1),(-4,4))   # bounds on variables
    print("Optimizing...")
    res = optimize.minimize(negLogLikelihood2, initial_guess, (xi_set, WSingleTag, True, normalizationAngles[0:1000000]), tol=10**-4, bounds=bnds)
    # '''
    print(res)




############################
# !!! WORK IN PROGRESS !!! #
############################
def MCintegral(par, uniformAngles):
    s = 0   # sum
    n = 0   # number of points
    for point in uniformAngles: # point is a 2D list
        s += WSingleTag(*par, point)
        n += 1
    return 1/n * s * (2*math.pi)**2    # MC-integral

# Generalized LL-func.: send in a pdf too, and let par be n-dim, dataset var X be m-dim.
# NOTE: Note working.
def negLogLikelihood2(par, var, pdf, normalizeSeparately=False, normalizationAngles=[]):
    '''Minimize this function for decay parameters to find max of Log-Likelihood for distribution. \n
    par : decay parameters to maximize [list], N-dim \n
    var : dataset of variables (xi) [list of lists] M-dim (NOTE: the inner lists represent observed points, i.e. 
    every variable is not a separate list, but rather part of a set of variables (i.e. a point)). 
    E.g.:
    >>> var = [ [a0, b0], [a1, b1], [a2, b2], ... ] and not \n
    >>> var = [ [a0, a1, a2, ...], [b0, b1, b2, ...] ] 
    where a, b are different variables for the pdf. \n
    pdf : must take arguments pdf(p1,p2, ..., pN, v1, v2, ..., vM)'''
    
    print("Hello!")
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

################3

if __name__ == "__main__":
    main()
