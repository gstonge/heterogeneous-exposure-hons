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

#plot
width = 7.057
height = width/3.7
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.2, bottom=0.24, right=0.85,
                    top=0.98, wspace=0.48, hspace=0.25)

color_dict = {1.1:"#c7e9b4", 1.15:"#7fcdbb", 1.5:"#0c2c84"}

#left panel
#-------------
alpha_list = [key for key in color_dict]

with open('./dat/bifurcation_mean_rescale_upper.pk', 'rb') as filename:
    results = pickle.load(filename)
for alpha in alpha_list:
    axes[0].plot(results[alpha]['beta'],results[alpha]['I'], '-',
                 color=color_dict[alpha], zorder=1, label=fr"$\alpha={alpha}$")
for alpha in (1.5,1.1,1.15):
    axes[0].plot((0.14,results[alpha]['beta_c']),(0,0), '-',
                color=color_dict[alpha], zorder=1)
    if alpha == 1.5:
        axes[0].vlines(results[alpha]['beta_c'],0,0.455,
                        ls='--',color=color_dict[alpha])
        axes[0].vlines(results[alpha]['beta'][-1],0,
                       results[alpha]['I'][-1],
                        ls='--',color=color_dict[alpha])

axes[0].legend(frameon=False, loc='lower right')
axes[0].set_xlabel(r"$\beta$")
axes[0].set_ylabel(r"Stationary prevalence $I^*$")
axes[0].text(0.1, 0.85, r'(a)', fontsize=font_size+1,
             transform=axes[0].transAxes)

#right panel
#-------------
with open('./dat/invasion_threshold_mean_rescaled.pk', 'rb') as filename:
    result = pickle.load(filename)

ind = np.argmin(result['betac'])
axes[1].plot(result['alpha'][:],result['betac'][:], color="#1a1a1a")
axes[1].set_xlabel(r"$\alpha$")
axes[1].set_ylabel(r"Invasion threshold $\beta_\mathrm{c}$")
style = "Simple, tail_width=0.5, head_width=3, head_length=6"
kw = dict(arrowstyle=style, color="#1a1a1a")
a1 = patches.FancyArrowPatch((1.3, 0.4), (1.18, 0.15),
                             connectionstyle="arc3,rad=0.15",**kw)
axes[1].add_patch(a1)
axes[1].text(1.305, 0.405, r'$\alpha^*$',
             fontsize=font_size+1)
axes[1].text(0.1, 0.85, r'(b)', fontsize=font_size+1,
             transform=axes[1].transAxes)
# axes[1].set_ylim([0.1401,0.188])

plt.savefig('./figs/FigS5.pdf')
plt.show()
