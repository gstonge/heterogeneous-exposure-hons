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
nb_graph = 10
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
betalist = np.linspace(0.1,0.3,50)
K = 1
initial_infected_fraction = 0.5
nb_history = 50
integrand = exponential_integrand

#results
results = dict()
results['simulation'] = {'Ilist_avg':[], 'Ilist_std':[]}
results['beta'] = betalist
results['ame'] = []
results['binomial'] = []

for j,beta in enumerate(betalist):
    prevalence = []
    for i in range(nb_graph):
        klist = horgg.utility.sequence_1(N, pk)
        mlist = horgg.utility.sequence_2(klist,pm)

        graph_generator = horgg.BCMS(klist,mlist)
        horgg.BCMS.seed(seed+i+j*nb_graph) #optional, if nothing is given, it is seeded with time

        edge_list = graph_generator.get_random_graph(nb_steps=int(np.sum(klist)))

        cont = HeterogeneousExposure(edge_list,recovery_probability,alpha,T,beta,K)
        cont.seed(seed+i+j*nb_graph) #optional
        cont.infect_fraction(initial_infected_fraction)
        cont.initialize_history(nb_history)

        #define some measures
        cont.measure_prevalence()

        #burnin
        dt = 1000
        dec_dt = 1
        cont.evolve(dt,dec_dt,measure=False,quasistationary=True)

        #evolve and measure
        dt = 1000
        dec_dt = 10
        cont.evolve(dt,dec_dt,measure=True,quasistationary=True)

        #print the result measure
        for measure in cont.get_measure_vector():
            name = measure.get_name()
            if name == "prevalence":
                prevalence.extend(measure.get_result())

    results['simulation']['Ilist_avg'].append(np.mean(prevalence))
    results['simulation']['Ilist_std'].append(np.std(prevalence))
    print(f"beta {beta}, I: {results['simulation']['Ilist_avg'][-1]}")
plt.errorbar(betalist,results['simulation']['Ilist_avg'],
             yerr=results['simulation']['Ilist_std'])

#binomial model
for beta in betalist:
    Ik = initial_infected_fraction*np.ones(kvec.shape)
    thetami = binomial_model.get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=1,T=T)
    Ilast = None
    Inow = np.sum(Ik*pk)
    while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)):
        Ik = binomial_model.evolution(Ik, pk, kvec, pm, mvec, thetami,
                           recovery_probability)
        Ilast = Inow
        Inow = np.sum(Ik*pk)
    results['binomial'].append(Inow)
plt.plot(betalist,results['binomial'],'-',color='orange')


#ame model
for beta in betalist:
    Ik = initial_infected_fraction*np.ones(kvec.shape)
    thetami = binomial_model.get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=1,T=T)
    Hmi = ame_model.initialize_Hmi(mmax,initial_infected_fraction)
    Ilast = None
    Inow = np.sum(Ik*pk)
    while Ilast is None or (np.abs(Inow - Ilast)/Inow > 10**(-8)):
        Hmi,Ik = ame_model.evolution(Hmi, Ik, pk, kvec, pm, mvec, thetami,
                           recovery_probability)
        Ilast = Inow
        Inow = np.sum(Ik*pk)
    results['ame'].append(Inow)
plt.plot(betalist,results['ame'],'-',color='blue')

plt.ylabel("Stationary prevalence $I^*$")
plt.xlabel(r"$\beta$")
plt.show()

with open('./dat/figureS7_phase_transition.pk','wb') as filename:
    pickle.dump(results,filename)
