import uproot
import numpy as np
from time import time


filename = "angleTree.root"

file = uproot.open(filename)   # >>> file.keys()            ['Angles;1']
# >>> file.classnames()         {'Angles': 'TTree'}
tree = file['Angles']

branches = tree.arrays(library='np')
cos_th = branches['cosThLam']
cos_thP = branches['cosThPn']
print(cos_th[:3])

t0 = time()
vars1 = np.vstack((cos_th,cos_thP)).T    # returns what I want          # as fast?
t1 = time()
print(f"One done, \t {t1-t0:.5f} secs")
vars2 = np.column_stack((cos_th,cos_thP))    # returns what I want as well
t2 = time()
print(f"second done, \t {t2-t1:.5f} secs")

print(vars2[:10])

'''
>>> file['Angles']
<TTree 'Angles' (2 branches) at 0x000007b9e730>

>>> tree.keys()
['cosThLam', 'cosThPn']

>>> tree.arrays(library="np")
{'cosThLam': array([ 0.9994835 ,  0.89440216,  0.91495391, ...,  0.08445024,
       -0.33016848, -0.04713729]), 'cosThPn': array([-0.67418025, -0.53668691,  0.48861069, ..., -0.82584422,
       -0.3210522 , -0.19814343])}
'''