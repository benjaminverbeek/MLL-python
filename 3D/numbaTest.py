import time
import numpy as np
import numba
from numba import jit
from numba.typed import List
import math


@jit(nopython=True)
def f(x,y):
    return x+y

@jit(nopython=True)
def doubleAdd(x,y):
    return f(x,y) + f(x,y)

print(doubleAdd(5,10))
# Read data
print("Fetching data...")
start_time = time.time()
normalizationAngles = [ list(map(float,i.split())) for i in open("lPHSP_4Pi_testing.txt").readlines() ]    # list (of lists)
normalizationAngles = np.asarray(normalizationAngles) # convet to NumPy array.
print(normalizationAngles[0])
print(f"Number of random angles for normalization: {len(normalizationAngles)}")
# NOTE: The normalization angles are not angles but rather cos(angles).
print("--- %s seconds ---" % (time.time() - start_time))


#####
alpha = 0.753   # assumed.
F0 = lambda cos_th, cos_thP : 1
F1 = lambda cos_th, cos_thP : math.sin(math.acos(cos_th)) * cos_th * cos_thP
F2 = lambda cos_th, cos_thP : (cos_th)**2

@jit(nopython=True)
def WSingleTag(eta, delta_phi, xi):
    cos_th, cos_thP = xi
    '''Normalize this to get the PDF we want to optimize. Xi is the set of angles, packed as a list/tuple/...'''
    return 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP  # W function

def MCintegralPy(par, uniformAngles):
    eta, delta_phi = par
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # point is a 2D list
        cos_th, cos_thP = xi
        s += 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP
        n += 1
    return 1/n * s * (2)**2    # MC-integral

@jit(nopython=True) # nopython=True doesn't appear to make a difference but is apparently recommended.
def MCintegralNum(par, uniformAngles):
    eta, delta_phi = par
    s = 0.0   # sum
    n = 0.0   # number of points
    for xi in uniformAngles: # point is a 2D list
        cos_th, cos_thP = xi
        s += 1 + eta * (cos_th)**2 + alpha * (1 - eta**2)**(0.5) * np.sin(delta_phi) \
            * np.sin(np.arccos(cos_th)) * cos_th * cos_thP
        n += 1
    return 1/n * s * (2)**2    # MC-integral
#####
'''
print("ok0")
t1 = time.time()
res1 = MCintegral([0.2, 42*math.pi/180], normalizationAngles[0:-1])
t2 = time.time()
print("ok1")
res2 = MCintegral([0.2, 42*math.pi/180], normalizationAngles[0:-1])
t3 = time.time()
print("ok2")
res3 = MCintegral([0.2, 52*math.pi/180], normalizationAngles[0:-1])
t4 = time.time()
print("ok3")
print("Some results: ", res1,res2, res3)

print(f"First iter took {t2-t1:.5f} s, second iter {t3-t2:.5f}, third iter {t4-t3:.5f}")
'''
print("Numba")
timesNumba = []
params = [0.2, 52*math.pi/180]  # eta, dPhi
typed_params = List()   # conversion needed for numba handling. I think unpacking breaks things.
[typed_params.append(x) for x in params]    # converts. Lists won't be supported later.
for i in range(5):
    t1 = time.time()
    res = MCintegralNum(typed_params, normalizationAngles[0:-1])
    t2 = time.time()
    timesNumba.append(t2-t1)
    print(t2-t1, " s")
    print(f"Value: {res}")

print("Py")
timesPy = []
for i in range(2):
    t1 = time.time()
    res = MCintegralPy([0.2, 52*math.pi/180], normalizationAngles[0:-1])
    t2 = time.time()
    timesPy.append(t2-t1)
    print(t2-t1, " s")
    print(f"Value: {res}")

aveNum = sum(timesNumba) / len(timesNumba)
avePy = sum(timesPy) / len(timesPy)
print(f"Results! average time Numba: {aveNum}, \t average time Python: {avePy}")