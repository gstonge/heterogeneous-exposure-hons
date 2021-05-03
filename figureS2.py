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

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(width,height))

plt.subplots_adjust(left=0.09, bottom=0.23, right=0.99,
                    top=0.90, wspace=0.28, hspace=0.25)


color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

integrand = gamma_special_integrand
tmin = 1
K = 1
beta = 0.05
alpha_list = [alpha for alpha in color_dict]
z_list = [0.5,0.8,1.]
rho_list = np.logspace(-2,0,100)
panel_label_list = ['(a)','(b)','(c)']

for col,(z,panel_label) in enumerate(zip(z_list,panel_label_list)):

    for alpha in alpha_list:
        label = fr"$\alpha = {alpha}$"
        kernel_list = [kernel(rho,beta,K=K,alpha=alpha,tmin=tmin,
                              integrand=integrand,args=(z,))
                  for rho in rho_list]

        axes[col].loglog(rho_list,kernel_list, '-', color=color_dict[alpha],
                          label=label)
    if col == 2:
        axes[col].loglog(rho_list,0.01*rho_list, '--', color="#1a1a1a",
                        label=r"$\propto \rho$")
    labelLines(list(axes[col].get_lines()), zorder=2.5,align=True,
                   color='black',fontsize=font_size-1)
    axes[col].set_xlabel(r"Environment prevalence $\rho$")
    axes[col].text(0.01, 1.05, fr'{panel_label} $z = {z}$',
                 fontsize=font_size+1,
                 transform=axes[col].transAxes)
    if col == 0:
        axes[col].set_ylabel(r"Infection probability $\theta_m$")


plt.savefig(f'../manuscript/figs/FigS2.pdf')
plt.show()
