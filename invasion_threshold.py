"""
Obtain the invasion threshold
"""
import numpy as np
from scipy.special import gamma, loggamma
from scipy.integrate import quad
from scipy.optimize import brentq
from new_kernel import *

#NOTE the function below overestimate the invasion threshold when
#alpha < 1. This is related to numerical instability when integrating
#to obtain the kernel. Changing the epsrel and epsabs to lower values seems to
#help, despite some warning.
def invasion_threshold(pm, pk, mu = 0.05, f = lambda m: 1,
                       betamin=10**(-8),betamax=2, **kernel_kwargs):
    kvec = np.arange(0,len(pk))
    kmean = np.sum(pk*kvec)
    k2mean = np.sum(pk*kvec**2)
    mvec = np.arange(0,len(pm))
    mmean = np.sum(pm*mvec)
    #function to find the root
    def func(beta):
        thetam = np.zeros(pm.shape)
        for m in range(2,len(pm)):
            rho = 1/(m-1)
            fm = f(m)
            thetam[m] = kernel(rho,beta,fm=fm,**kernel_kwargs)
        return np.sum(mvec*(mvec-1)*thetam*pm)/mmean - mu*kmean/k2mean
    return brentq(func,betamin,betamax)

