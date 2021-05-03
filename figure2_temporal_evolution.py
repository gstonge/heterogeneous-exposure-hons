import pickle
import matplotlib.pyplot as plt
from binomial_model import *
from scipy.special import loggamma
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
alpha_list = [0.5,1.,1.5]
beta_list = [0.0005,0.025,0.08]
integrand = exponential_integrand

result = dict()
for alpha,beta in zip(alpha_list,beta_list):
    result[alpha] = dict()
    result[alpha]['t_med'] = 2**(1/alpha)
    result[alpha]['t'] = []
    result[alpha]['I'] = []
    beta_c = invasion_threshold(pm, pk, mu, f,
                                K=K,alpha=alpha,tmin=tmin,T=T,
                                integrand=integrand)
    result[alpha]['beta_c'] = beta_c
    result[alpha]['beta'] = beta
    print(alpha, beta_c)

for alpha,beta in zip(alpha_list,beta_list):
    Ik = 0.0005*np.ones(kvec.shape)
    thetami = get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=tmin,T=T)
    Ilast = None
    Inow = np.sum(Ik*pk)
    result[alpha]['I'].append(Inow)
    result[alpha]['t'].append(0)
    # while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)):
    for _ in range(int(1000/2**(1/alpha))):
        Ik = evolution(Ik, pk, kvec, pm, mvec, thetami, mu)
        Ilast = Inow
        Inow = np.sum(Ik*pk)
        result[alpha]['I'].append(Inow)
        result[alpha]['t'].append(result[alpha]['t'][-1]+1)
    print(f"alpha {alpha}, I : {Inow}")

    plt.semilogy(np.array(result[alpha]['t'][1:])*result[alpha]['t_med'],
                 result[alpha]['I'][1:], label=f'alpha = {alpha}')
plt.legend()
plt.show()

with open('./dat/figure2_temporal_evolution.pk','wb') as filename:
    pickle.dump(result,filename)
