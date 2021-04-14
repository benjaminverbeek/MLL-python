from HitAndMissGenerator import hit_miss_generator
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import time

# probability density function = func / (integral of func w.r.t. variables X), i.e. normalize integral.
# pdf(x,alpha) now yields the probability of seeing x given a distribution with parameter alpha.
# int [-1,1] of 1+ax^2 dx = 2*(x + ax^3/3) | x=1 = (2*3 + 2*a)/3
pdf = lambda costheta, alpha : (3 /(6+2*alpha)) * (1 + alpha*costheta**2) # normalized to integrate to 1 on [-1,1]

# what we want to maximize... (minimize neg.)
def negLogLikelihoodFunc(alpha, X):
    '''Dataset X [list], parameter alpha [float]'''
    s = 0       # sum
    for i in X: # iterate over samples
        s -= np.log(pdf(i,alpha)) # log-sum of pdf gives LL
    return s

def main():
    
    start_time = time.time()

    # Set some parameters for the event generator
    N = 10000   # number of events to be generated
    alpha = -0.2 # ratio between G_E and G_M

    print(f"Generating {N} random points...")

    costheta_dist = []
    for i in hit_miss_generator(N, alpha):
        costheta_dist.append(i)

    print('Done generating.')
    cp1 = time.time()
    print(f"--- {cp1 - start_time} seconds since start---")

    print("Minimizing negLogLikelihoodFunc...")
    # NOTE: This can easily be made multivalued. Just pack extra arguments in args-parameter (as many as needed)
    # and pack variables to optimize for aswell (make sure initial guess is same dimension as variables)...
    res = optimize.minimize(negLogLikelihoodFunc, -0.3, costheta_dist, tol=10**-3)

    print(f"Deviation from alpha = {alpha}: {res['x'][0]-(alpha)}") # depends on set of points obviously.
    print(res)
    print(f'Convergence: {res["success"]}')

    cp2 = time.time()
    print(f"--- {cp2 - cp1} seconds since start last checkpoint ---")
    print(f"--- {cp2 - start_time} seconds since start---")




if __name__ == "__main__":
    main()