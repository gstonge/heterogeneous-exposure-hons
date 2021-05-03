from invasion_threshold import *
import pickle

def poisson(xvec, xmean):
    return np.exp(xvec*np.log(xmean)-xmean-loggamma(xvec+1))

alpha_list = np.linspace(1.01,3.5,100)
betac_list = []
for alpha in alpha_list:
    tmin = (alpha-1)/alpha
    T = np.inf
    f = lambda m: 1
    K = 1
    integrand = exponential_integrand

    mu = 0.05
    mmax = 40
    kmax = 20
    mmean = 10
    kmean = 5
    mvec = np.arange(mmax+1)
    kvec = np.arange(kmax+1)
    pm = poisson(mvec,mmean)
    pk = poisson(kvec,kmean)

    betac_list.append(invasion_threshold(pm, pk, mu, f,
                             K=K,alpha=alpha,tmin=tmin,T=T,
                                         integrand=integrand))
    print(f"alpha {alpha}: ",betac_list[-1])
result = {'alpha':alpha_list, 'betac':betac_list}
with open('./dat/invasion_threshold_mean_rescaled.pk', 'wb') as filename:
    pickle.dump(result,filename)

