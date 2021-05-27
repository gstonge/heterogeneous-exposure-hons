import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import matplotlib.patches as patches
from labellines import labelLine, labelLines
from new_kernel import *

#plot parameters
font_size=8
plt.style.use('seaborn-paper')
plt.rc('text', usetex=True)
plt.rc('font',family='serif',serif='Computer Modern')
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('axes', labelsize=font_size)
plt.rc('legend', fontsize=font_size)

#color list
color_list = ["#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8", "#0c2c84"]
newcm = LinearSegmentedColormap.from_list('ColorMap',
                                          list(reversed(color_list[:-1])))

#plot
width = 7.057
height = width/4
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.09, bottom=0.22, right=0.98,
                    top=0.9, wspace=0.45, hspace=0.25)

color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

#left plot : kernel
#-------------------
alpha_list = [0.5,1.,1.5]
rho_list = np.logspace(-2,0,100)
beta = 0.1
fm = 1
K = 1
tmin = 1
T = np.inf
integrand=exponential_integrand

for alpha in alpha_list:
    label=fr"$\alpha = {alpha}$"
    kernel_list = [kernel(rho,beta,fm,K,alpha,tmin,T,integrand=integrand)
                   for rho in rho_list]
    axes[0].loglog(rho_list,kernel_list, '-', color=color_dict[alpha],
                  label=label)
labelLines(list(axes[0].get_lines()), zorder=2.5,align=True,
           color='black',fontsize=font_size-1)
axes[0].set_xlabel(r"Environment prevalence $\rho$")
axes[0].set_ylabel(r"Infection probability $\theta_m$")
axes[0].text(0.02, 1.07, r'(a) Infection kernel', fontsize=font_size+1,
             transform=axes[0].transAxes)
axes[0].text(0.5, 0.15, r'$\theta_m(\rho) \propto \rho^\alpha$',
             fontsize=font_size,
             transform=axes[0].transAxes)

mt = [i*10**(-j) for i in range(2,10) for j in range(1,6)]
axes[0].set_yticks(mt, minor=True)
axes[0].set_yticks([10**(-1),10**(-2),10**(-3),10**(-4)], minor=False)
axes[0].set_ylim([2*10**(-5),5*10**(-1)])


#center plot : temporal evo
#---------------------------
with open('./dat/figure2_temporal_evolution.pk', 'rb') as filename:
    result = pickle.load(filename)

for alpha in alpha_list:
    t = np.array(result[alpha]['t'])
    t_med = result[alpha]['t_med']
    I = result[alpha]['I']
    label=fr"$\alpha = {alpha}$"
    if alpha == 1.:
        zorder = -1
    else:
        zorder = 2
    axes[1].semilogy(t*t_med,I, '-', color=color_dict[alpha],
                  label=label, zorder=zorder)

#get exponential rate
ind = 5
t_med = result[0.5]['t_med']
rate = np.log(result[0.5]['I'][ind+1]) - np.log(result[0.5]['I'][ind])
line = np.exp(rate*t[0:21])*2*10**(-3)
axes[1].semilogy(t[0:21]*t_med-20,line, ':', color="#1a1a1a",
              label=label)
ind = 100
t_med = result[1.5]['t_med']
rate = np.log(result[1.5]['I'][ind+1]) - np.log(result[1.5]['I'][ind])
line = np.exp(rate*t[ind:600])*3*10**(-4)
axes[1].semilogy(t[ind:600]*t_med,line, '--', color="#1a1a1a",
              label=label)
axes[1].set_xlabel(r"Time $t \times \bar{\tau}$")
axes[1].set_ylabel(r"Prevalence $I(t)$")

axes[1].text(0.02, 1.07, r'(b) Temporal evolution', fontsize=font_size+1,
             transform=axes[1].transAxes)

#right plot : bifurcation
#-------------------------
with open('./dat/figure2_bifurcation.pk', 'rb') as filename:
    results = pickle.load(filename)

for alpha in reversed(alpha_list):
    axes[2].plot(results[alpha]['beta'],results[alpha]['I'], '-',
                 color=color_dict[alpha])
    if alpha != 0.5:
        axes[2].plot((0.,results[alpha]['beta_c']),(0,0), '-',
                    color=color_dict[alpha], zorder=1)
    if alpha == 1.5:
        axes[2].vlines(results[alpha]['beta_c'],0,0.455,
                        ls='--',color="#1a1a1a")
        axes[2].vlines(results[alpha]['beta'][-1],0,
                       results[alpha]['I'][-1],
                       ls=':',color="#1a1a1a")


style = "Simple, tail_width=0.5, head_width=3, head_length=6"
kw = dict(arrowstyle=style, color="#1a1a1a")
a1 = patches.FancyArrowPatch((0.058, 0.35), (0.054, 0.05),
                             connectionstyle="arc3,rad=0.15",**kw)
axes[2].add_patch(a1)

a2 = patches.FancyArrowPatch((0.08, 0.05), (0.08, 0.35),
                             connectionstyle="arc3,rad=0.",**kw)
axes[2].add_patch(a2)

axes[2].set_xlabel(r"$\beta$")
axes[2].set_xticks([0,0.05,0.10,0.15])
axes[2].set_yticks([0,0.2,0.4,0.6,0.8])
axes[2].set_ylabel(r"Stationary prevalence $I^*$")

axes[2].text(0.02, 1.07, r'(c) Phase diagram', fontsize=font_size+1,
             transform=axes[2].transAxes)


plt.savefig('figs/Fig2.pdf')
plt.show()


