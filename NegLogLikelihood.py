# A generalized LL-function which can be minimized to find decay parameters par given a
# dataset of points for xi (angles). PDF must be defined and sent in separately.

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