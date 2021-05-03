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
T = np.inf
mmax = 40
kmax = 20
mmean = 10
kmean = 5
mvec = np.arange(mmax+1)
kvec = np.arange(kmax+1)
pm = poisson(mvec,mmean)
pk = poisson(kvec,kmean)
alpha_list = [1.1,1.15,1.5]
integrand = exponential_integrand

beta_list = np.linspace(0.34,0.12,500)

result = dict()
for alpha in alpha_list:
    tmin = (alpha-1)/alpha
    result[alpha] = dict()
    result[alpha]['beta'] = []
    result[alpha]['I'] = []
    beta_c = invasion_threshold(pm, pk, mu, f,
                                K=K,alpha=alpha,tmin=tmin,T=T,
                                integrand=integrand)
    result[alpha]['beta_c'] = beta_c

for alpha in alpha_list:
    tmin = (alpha-1)/alpha
    Ik = 0.9*np.ones(kvec.shape)
    print(f"----------------")
    print(f"alpha {alpha}")
    print(f"----------------")
    for beta in beta_list:
        thetami = get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=tmin,T=T)
        Ilast = None
        Inow = np.sum(Ik*pk)
        while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-6)
                                and Inow > 10**(-4)):
            Ik = evolution(Ik, pk, kvec, pm, mvec, thetami, mu)
            Ilast = Inow
            Inow = np.sum(Ik*pk)
        print(f"beta {beta}, I : {Inow}")
        if Inow <= 10**(-4):
            break
        result[alpha]['I'].append(Inow)
        result[alpha]['beta'].append(beta)

    plt.plot(result[alpha]['beta'],result[alpha]['I'],label=f'alpha = {alpha}')
plt.legend()
plt.show()

with open('./dat/bifurcation_mean_rescale_upper.pk','wb') as filename:
    pickle.dump(result,filename)
