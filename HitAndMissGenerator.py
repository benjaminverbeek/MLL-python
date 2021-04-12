# Import the any necessary modules
import numpy as np
import matplotlib.pyplot as plt


# Set the seed for the random number generator
np.random.seed()


# Set some parameters for the event generator
N = 1000000   # number of events to be generated
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
pdf = lambda costheta, alpha : 3* (1 + alpha*costheta**2)/(6+2*alpha) # normalized to integrate to 1 on [-1,1]

step = 0.05
a = [-1+step*i for i in range(int(1//step))] # range of alphas
#a = [-0.3, -0.2, -0.1] # guesses for alpha

# this is a terrible brute-force solution. I should check zeroes of derivative (numerically or analytically).
# takes quite a while... 
L = [] # log-likelihood. We want to maximize this.
for alph in a: # for each alpha, check the log-likelihood of seeing all generated points.
  p = 0
  for i in costheta_dist:
      p += np.log(pdf(i,alph))
  L.append(p)

#print(a)
print(L)
ind = L.index(max(L))
print(f"Max at alpha = {a[ind]:.2f}, yields L = {max(L)}")
# note to self: f"txt {x:totalChars.decimalPrecision f}" formats floats. use totalChars for "tabs"