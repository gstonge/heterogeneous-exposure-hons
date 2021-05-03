import pickle
import matplotlib.pyplot as plt
from ame_model import *
from new_kernel import *
from scipy.special import loggamma

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
integrand = exponential_integrand

beta_list = np.linspace(0.15,0.000011,500)

result = dict()
for alpha in alpha_list:
    result[alpha] = dict()
    result[alpha]['beta'] = []
    result[alpha]['I'] = []

for alpha in alpha_list:
    epsilon = 0.9
    Ik = epsilon*np.ones(kvec.shape)
    Hmi = initialize_Hmi(mmax,epsilon)
    print(f"----------------")
    print(f"alpha {alpha}")
    print(f"----------------")
    for beta in beta_list:
        thetami = get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=tmin,T=T)
        Ilast = None
        Inow = np.sum(Ik*pk)
        while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)
                                and Inow > 10**(-4)):
            Hmi,Ik = evolution(Hmi, Ik, pk, kvec, pm, mvec, thetami, mu)
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

with open('./dat/figureS6_bifurcation.pk','wb') as filename:
    pickle.dump(result,filename)
