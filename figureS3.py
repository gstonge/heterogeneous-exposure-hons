import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
height = width/3.3

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.23, bottom=0.23, right=0.87,
                    top=0.90, wspace=0.28, hspace=0.25)


color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

integrand = exponential_integrand
tmin = 1
K = 1
alpha_list = [alpha for alpha in color_dict]
rho_list = np.logspace(-2,0,100)

#first case, tau_c \sim 1 for rho < 0.1
beta = 1
T = np.inf

for alpha in alpha_list:
    label = fr"$\alpha = {alpha}$"
    kernel_list = [kernel(rho,beta,K=K,alpha=alpha,tmin=tmin,T=T,
                          integrand=integrand)
              for rho in rho_list]

    axes[0].loglog(rho_list,kernel_list, '-', color=color_dict[alpha],
                      label=label)
axes[0].legend(frameon=False)
axes[0].set_xlabel(r"Environment prevalence $\rho$")
axes[0].text(0.01, 1.05, r'(a) $1 \approx \tau_\mathrm{c} \ll \mathcal{T}$',
             fontsize=font_size+1,
             transform=axes[0].transAxes)
axes[0].set_ylabel(r"Infection probability $\theta_m$")
axes[0].axvline(0.1, ls='--', color='#1a1a1a')

#second case, T \approx tau_c

beta = 0.1
T = 10**3

for alpha in alpha_list:
    label = fr"$\alpha = {alpha}$"
    kernel_list = [kernel(rho,beta,K=K,alpha=alpha,tmin=tmin,T=T,
                          integrand=integrand)
              for rho in rho_list]

    axes[1].loglog(rho_list,kernel_list, '-', color=color_dict[alpha],
                      label=label)
axes[1].legend(frameon=False)
axes[1].set_xlabel(r"Environment prevalence $\rho$")
axes[1].axvline(0.1, ls='--', color='#1a1a1a')
axes[1].text(0.01, 1.05, r'(b) $1 \ll \tau_\mathrm{c} \approx \mathcal{T}$',
             fontsize=font_size+1,
             transform=axes[1].transAxes)

plt.savefig(f'../manuscript/figs/FigS3.pdf')
plt.show()
