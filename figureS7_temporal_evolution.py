import horgg
import pickle
import matplotlib.pyplot as plt
import numpy as np
from _schon import HeterogeneousExposure
import ame_model
import binomial_model
from invasion_threshold import *
from scipy.special import loggamma

def poisson(xvec, xmean):
    return np.exp(xvec*np.log(xmean)-xmean-loggamma(xvec+1))

seed = 42

#structure
nb_graph = 200
N = 10000
mmax = 40
kmax = 20
mmean = 10
kmean = 5
mvec = np.arange(mmax+1)
kvec = np.arange(kmax+1)
pm = poisson(mvec,mmean)
pk = poisson(kvec,kmean)


#infection parameter
recovery_probability = 0.05
alpha = 2.
T = np.inf
beta = 0.2
K = 1
initial_infected_fraction = 0.01
nb_history = 50
integrand = exponential_integrand

#results
results = dict()
results['simulation'] = {'Idata':[]}
results['binomial'] = dict()
results['ame'] = dict()

for i in range(nb_graph):
    klist = horgg.utility.sequence_1(N, pk)
    mlist = horgg.utility.sequence_2(klist,pm)

    graph_generator = horgg.BCMS(klist,mlist)
    horgg.BCMS.seed(seed+i) #optional, if nothing is given, it is seeded with time

    edge_list = graph_generator.get_graph(mcmc_step=N)

    cont = HeterogeneousExposure(edge_list,recovery_probability,alpha,T,beta,K)
    cont.seed(seed+i) #optional
    cont.infect_fraction(initial_infected_fraction)
    cont.initialize_history(nb_history)

    #define some measures
    cont.measure_prevalence()


    #evolve and measure
    dt = 250
    dec_dt = 1
    cont.evolve(dt,dec_dt,measure=True,quasistationary=False)

    #print the result measure
    for measure in cont.get_measure_vector():
        name = measure.get_name()
        if name == "prevalence":
            prevalence = [initial_infected_fraction] + measure.get_result()

    results['simulation']['Idata'].append(prevalence)
    results['simulation']['tlist'] = np.arange(0,dt)
    plt.semilogy(np.arange(0,dt),prevalence, alpha = 0.1, color="#1a1a1a")

#binomial model
Ik = initial_infected_fraction*np.ones(kvec.shape)
thetami = binomial_model.get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=1,T=T)
Ilast = None
Inow = np.sum(Ik*pk)
Ilist = []
tlist = []
Ilist.append(Inow)
tlist.append(0)
# while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)):
for _ in range(250):
    Ik = binomial_model.evolution(Ik, pk, kvec, pm, mvec, thetami,
                       recovery_probability)
    Ilast = Inow
    Inow = np.sum(Ik*pk)
    Ilist.append(Inow)
    tlist.append(tlist[-1]+1)
results['binomial']['tlist'] = tlist
results['binomial']['Ilist'] = Ilist
plt.semilogy(tlist,Ilist,color='orange')


#ame model
Ik = initial_infected_fraction*np.ones(kvec.shape)
Hmi = ame_model.initialize_Hmi(mmax,initial_infected_fraction)
Ilast = None
Inow = np.sum(Ik*pk)
Ilist = []
tlist = []
Ilist.append(Inow)
tlist.append(0)
# while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)):
for i in range(250):
    Hmi,Ik = ame_model.evolution(Hmi, Ik, pk, kvec, pm, mvec, thetami,
                       recovery_probability)
    Ilast = Inow
    Inow = np.sum(Ik*pk)
    Ilist.append(Inow)
    tlist.append(tlist[-1]+1)
results['ame']['tlist'] = tlist
results['ame']['Ilist'] = Ilist
plt.semilogy(tlist,Ilist,color='blue')

plt.ylabel("prevalence")
plt.xlabel("time")
plt.show()

with open('./dat/figureS7_temporal_evolution.pk','wb') as filename:
    pickle.dump(results,filename)
