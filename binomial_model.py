import numpy as np
from scipy.integrate import quad
from scipy.special import binom
from scipy.special import gamma
from scipy.special import gammainc
from scipy.stats import binom
from numba import njit

def get_thetami_mat(mmax, beta, f=lambda m: 1, K=1, alpha=2, tmin=1, T=np.inf):
    #uses an exponential dose distribution
    #thetami = thetam(i/m-1)
    thetami = np.zeros((mmax+1,mmax+1))
    Z = (tmin**(-alpha)-T**(-alpha))/alpha
    for m in range(2,mmax+1):
        for i in range(1,m):
            tau_c = K*(m-1)/(beta*i*f(m))
            thetami[m,i] += np.exp(-tau_c)-np.exp(-tau_c/T)*T**(-alpha)
            thetami[m,i] += tau_c**(-alpha)*gamma(alpha+1)*(\
                                    gammainc(alpha+1,tau_c)-
                                    gammainc(alpha+1,tau_c/T))
    return thetami/(Z*alpha)

@njit
def get_binom(N,p):
    if N > 0:
        pmf = np.zeros(N+1)
        pmf[0] = (1-p)**N
        for i in range(N):
            pmf[i+1] = pmf[i]*(N-i)*p/((i+1)*(1-p))
    else:
        pmf = np.array([1.])
    return pmf

@njit
def get_thetam_bar(rho_bar,thetami,mvec,mmax):
    thetam_bar = np.zeros(mmax+1)
    for m in range(2,mmax+1):
        thetam_bar[m] = np.sum(thetami[m,0:m]*get_binom(m-1,rho_bar))
    return thetam_bar

@njit
def get_theta_bar(thetam_bar,pm,mvec):
    return np.sum(mvec*thetam_bar*pm)/np.sum(mvec*pm)

@njit
def get_rho_bar(Ik,pk,kvec):
    return np.sum(kvec*Ik*pk)/np.sum(kvec*pk)

@njit
def get_I(Ik,pk):
    return np.sum(pk*Ik)

@njit
def evolution(Ik, pk, kvec, pm, mvec, thetami, mu):
    mmax = mvec[-1]
    rho_bar = get_rho_bar(Ik,pk,kvec)
    thetam_bar = get_thetam_bar(rho_bar,thetami,mvec,mmax)
    theta_bar = get_theta_bar(thetam_bar,pm,mvec)
    Thetak = 1 - (1-theta_bar)**kvec
    return (1-mu)*Ik + (1-Ik)*Thetak


if __name__ == '__main__':
    from invasion_threshold import *

    def poisson(xvec, xmean):
        return np.exp(xvec*np.log(xmean)-xmean-loggamma(xvec+1))

    #parameter
    mu = 0.05
    f = lambda m: 1
    K = 1
    tmin = 1
    T = np.inf
    mmax = 40
    kmax = 20
    mmean = 10
    kmean = 5
    mvec = np.arange(mmax+1)
    kvec = np.arange(kmax+1)
    pm = poisson(mvec,mmean)
    pk = poisson(kvec,kmean)
    alpha = 0.5
    integrand = exponential_integrand

    #estimate invasion threshold
    beta_c = invasion_threshold(pm, pk, mu, f,
                                K=K,alpha=alpha,tmin=tmin,T=T,
                                integrand=exponential_integrand)
    print(f"invasion threshold : {beta_c}")


    beta = 0.0152
    thetami = get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=tmin,T=T)

    epsilon = 10**(-10)
    Ik = epsilon*np.ones(kvec.shape)
    print("----------------------")
    print("temporal evolution")
    print("----------------------")
    for t in range(1000):
        Ik = evolution(Ik, pk, kvec, pm, mvec, thetami, mu)
        if (t % 20) == 0:
            print(np.sum(Ik*pk))
    print(np.sum(Ik*pk))

