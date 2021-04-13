# Import the any necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# Set the seed for the random number generator
np.random.seed()


# Set some parameters for the event generator
N = 100000   # number of events to be generated
alpha = -0.2 # ratio between G_E and G_M
bins = 100 # number of bins for the histogram


# Implement the hit or miss generator
# generates distribution following 1 + alpha * x^2
def hit_miss_generator(N, alpha):
  i = 0
  while (i < N):
    costheta = np.random.uniform(-1, 1)
    rand = np.random.uniform(0, 1)
    dsdcostheta = 1 + alpha * costheta**2 # calculate the angular distribution
    upper_limit = 1 + abs(alpha) # and normalise it
    i += 1
    if (rand < dsdcostheta / upper_limit): # makes it follow distribution
      yield costheta

# Create and fill the cos(theta) distribution
costheta_dist = []
for i in hit_miss_generator(N, alpha):
  costheta_dist.append(i)

'''
# Let's look at the result
plt.hist(costheta_dist, bins)
plt.xlabel('$\cos(\\theta)$')
plt.ylabel('entries')
plt.show()
'''
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

# run optimization (really slow!): # TODO: select suitable method
res = optimize.minimize(negLogLikelihoodFunc, -0.5, costheta_dist, tol=10**-3)

print(f"Deviation from alpha = {alpha}: {res['x'][0]-(alpha)}") # depends on set of points obviously.
print(res)
print(f'Convergence: {res["success"]}')

'''
#print(a)
print(len(L))
ind = L.index(max(L))
print(f"Max at alpha = {a[ind]:.2f}, yields L = {max(L)}")
# note to self: f"txt {x:totalChars.decimalPrecision f}" formats floats. use totalChars for "tabs"
'''