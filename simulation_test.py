import matplotlib.pyplot as plt
import numpy as np
from _schon import HeterogeneousExposure, DiscreteSIS
from binomial_model import get_thetami_mat

#structure : all individuals belong to two groups
N = 500
edge_list = []
for node in range(N):
    edge_list.append((node,0))
    edge_list.append((node,1))

#infection parameter
recovery_probability = 0.05
alpha = 1.5
T = np.inf
beta = 0.14
K = 1
initial_infected_fraction = 0.5
seed = 42
nb_history = 50

cont = HeterogeneousExposure(edge_list,recovery_probability,alpha,T,beta,K)
cont.infect_fraction(initial_infected_fraction)
# cont.seed(seed) #optional
cont.initialize_history(nb_history)

#define some measures
cont.measure_prevalence()
cont.measure_marginal_infection_probability()

#evolve in the quasistationary state without measuring (burn-in)
dt = 10000
dec_dt = 10
cont.evolve(dt,dec_dt,measure=False,quasistationary=True)

#evolve and measure
dt = 10000
dec_dt = 10
cont.evolve(dt,dec_dt,measure=True,quasistationary=True)

#print the result measure
for measure in cont.get_measure_vector():
    name = measure.get_name()
    if name == "prevalence":
        print("----------------")
        print(name)
        print("----------------")
        print(np.mean(measure.get_result()))
    elif name == "marginal_infection_probability":
        plt.hist(measure.get_result())
        plt.show()

#now verification with other approach
infection_probability = get_thetami_mat(N, beta, f=lambda m: 1, K=K,
                                        alpha=alpha, tmin=1, T=T)
cont = DiscreteSIS(edge_list,recovery_probability,infection_probability)
cont.infect_fraction(initial_infected_fraction)
# cont.seed(seed) #optional
cont.initialize_history(nb_history)

#define some measures
cont.measure_prevalence()
cont.measure_marginal_infection_probability()

#evolve in the quasistationary state without measuring (burn-in)
dt = 10000
dec_dt = 10
cont.evolve(dt,dec_dt,measure=False,quasistationary=True)

#evolve and measure
dt = 10000
dec_dt = 10
cont.evolve(dt,dec_dt,measure=True,quasistationary=True)

#print the result measure
for measure in cont.get_measure_vector():
    name = measure.get_name()
    if name == "prevalence":
        print("----------------")
        print(name)
        print("----------------")
        print(np.mean(measure.get_result()))

    elif name == "marginal_infection_probability":
        plt.hist(measure.get_result())
        plt.show()

