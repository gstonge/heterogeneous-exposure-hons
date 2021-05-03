import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import matplotlib.patches as patches
from labellines import labelLine, labelLines
from new_kernel import *

#plot parameters
font_size=10
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
height = width/3.8
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.25, bottom=0.24, right=0.83,
                    top=0.88, wspace=0.45, hspace=0.25)

color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

#left plot : kernel
#-------------------
alpha_list = [0.5,1.,1.5]
with open('./dat/figureS6_temporal_evolution.pk', 'rb') as filename:
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
    axes[0].semilogy(t*t_med,I, '-', color=color_dict[alpha],
                  label=label, zorder=zorder)

#get exponential rate
ind = 5
t_med = result[0.5]['t_med']
rate = np.log(result[0.5]['I'][ind+1]) - np.log(result[0.5]['I'][ind])
line = np.exp(rate*t[0:21])*2*10**(-3)
axes[0].semilogy(t[0:21]*t_med-20,line, ':', color="#1a1a1a",
              label=label)
ind = 80
t_med = result[1.5]['t_med']
rate = np.log(result[1.5]['I'][ind+1]) - np.log(result[1.5]['I'][ind])
line = np.exp(rate*t[ind:500])*3*10**(-4)
axes[0].semilogy(t[ind:500]*t_med,line, '--', color="#1a1a1a",
              label=label)
axes[0].set_xlabel(r"Time $t \times \bar{\tau}$")
axes[0].set_ylabel(r"Prevalence $I(t)$")

axes[0].text(0.02, 1.05, r'(a) Temporal evolution', fontsize=font_size+1,
             transform=axes[0].transAxes)

#right plot : bifurcation
#-------------------------
with open('./dat/figureS6_bifurcation.pk', 'rb') as filename:
    results = pickle.load(filename)

for alpha in reversed(alpha_list):
    axes[1].plot(results[alpha]['beta'],results[alpha]['I'], '-',
                 color=color_dict[alpha])
    if alpha != 0.5:
        axes[1].plot((0.,results[alpha]['beta_c']),(0,0), '-',
                    color=color_dict[alpha], zorder=1)
    if alpha == 1.5:
        axes[1].vlines(results[alpha]['beta_c'],0,0.388,
                        ls='--',color=color_dict[alpha])
        axes[1].vlines(results[alpha]['beta'][-1],0,
                       results[alpha]['I'][-1],
                        ls='--',color=color_dict[alpha])

style = "Simple, tail_width=0.5, head_width=3, head_length=6"
kw = dict(arrowstyle=style, color="#1a1a1a")
a1 = patches.FancyArrowPatch((0.058, 0.35), (0.054, 0.05),
                             connectionstyle="arc3,rad=0.15",**kw)
axes[1].add_patch(a1)

a2 = patches.FancyArrowPatch((0.08, 0.05), (0.08, 0.35),
                             connectionstyle="arc3,rad=0.",**kw)
axes[1].add_patch(a2)

axes[1].set_xlabel(r"$\beta$")
axes[1].set_xticks([0,0.05,0.10,0.15])
axes[1].set_yticks([0,0.2,0.4,0.6,0.8])
axes[1].set_ylabel(r"Stationary prevalence $I^*$")

axes[1].text(0.02, 1.05, r'(b) Phase diagram', fontsize=font_size+1,
             transform=axes[1].transAxes)


plt.savefig('../manuscript/figs/FigS6.pdf')
plt.show()


