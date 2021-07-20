import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import matplotlib.patches as patches
from labellines import labelLine, labelLines

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


#left plot : kernel
#-------------------
with open('./dat/figureS7_temporal_evolution.pk', 'rb') as filename:
    results = pickle.load(filename)


tlist = results['simulation']['tlist']
nb_curves = 200
count = 0
for Ilist in results['simulation']['Idata']:
    count += 1
    if count == nb_curves:
        break
    axes[0].semilogy(tlist,Ilist, '-', color="#1a1a1a", alpha=0.1)

axes[0].set_xlabel(r"Time $t$")
axes[0].set_ylabel(r"Prevalence $I(t)$")

axes[0].semilogy(results['ame']['tlist'], results['ame']['Ilist'], '-',
                 color="#377eb8")
axes[0].semilogy(results['binomial']['tlist'], results['binomial']['Ilist'],
                 '--',color="#ff7f00")

axes[0].text(0.02, 1.05, r'(a) Temporal evolution', fontsize=font_size+1,
             transform=axes[0].transAxes)

#right plot : bifurcation
#-------------------------
with open('./dat/figureS7_phase_transition.pk', 'rb') as filename:
    results = pickle.load(filename)

axes[1].errorbar(results['beta'],results['simulation']["Ilist_avg"],
                 yerr=results['simulation']['Ilist_std'], fmt='o',
                 color="#1a1a1a",ms=4,alpha=0.5, zorder=-2)
axes[1].plot(results['beta'], results['ame'], '-',
            color="#377eb8",label="AME")
axes[1].plot(results['beta'], results['binomial'], '--',
            color="#ff7f00", label="Annealed")
leg = axes[1].legend()
leg.get_frame().set_linewidth(0.0)


axes[1].text(0.02, 1.05, r'(b) Phase diagram', fontsize=font_size+1,
             transform=axes[1].transAxes)

axes[1].set_xlabel(r"$\beta$")
axes[1].set_ylabel(r"Stationary prevalence $I^*$")

plt.savefig('./figs/FigS7.pdf')
plt.show()


