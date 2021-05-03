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
height = width/1.8

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(width,height))

plt.subplots_adjust(left=0.2, bottom=0.12, right=0.85,
                    top=0.92, wspace=0.48, hspace=0.25)


color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

#Beta distributions
pdf_list = [weibull_pdf,frechet_pdf]
integrand_list = [weibull_integrand,frechet_integrand]
shape_list = [2.,1.2]
tmin = 1
K = 1
kappa = np.linspace(0.001,5,1000)
beta = 0.1
alpha_list = [0.5,1.,1.5]


for row,(pdf,integrand,shape) in enumerate(zip(pdf_list,integrand_list,
                                              shape_list)):

    #left plot : dose distribution
    #-------------------
    pi = pdf(kappa, shape=shape)
    axes[row][0].plot(kappa,pi, color="#41b6c4")
    axes[row][0].fill_between(kappa,pi, color="#41b6c4", alpha=0.3)
    if row == 1:
        axes[row][0].set_xlabel(r"Dose $\kappa$")
    axes[row][0].set_ylabel(r"$\pi(\kappa)$")
    if row == 0:
        axes[row][0].text(0.02, 1.1, r'(a) \, Dose distribution',
                 fontsize=font_size+1,
                 transform=axes[row][0].transAxes)

    #right plot : kernel
    #-------------------
    rho_list = np.logspace(-2,0,100)


    for alpha in alpha_list:
        label = fr"$\alpha = {alpha}$"
        kernel_list = [kernel(rho,beta,K=K,alpha=alpha,tmin=tmin,
                              integrand=integrand, args=(shape,))
                  for rho in rho_list]

        axes[row][1].loglog(rho_list,kernel_list, '-', color=color_dict[alpha],
                          label=label)

    if row == 1:
        axes[row][1].loglog(rho_list,0.01*rho_list**(shape), '--', color="#1a1a1a",
                          label=fr"$\propto \rho^\psi$")
        axes[row][1].set_xlabel(r"Environment prevalence $\rho$")
    axes[row][1].set_ylabel(r"Infection probability $\theta_m$")
    if row == 0:
        axes[row][1].text(0.02, 1.1, r'(b) \, Infection kernel',
                 fontsize=font_size+1,
                 transform=axes[row][1].transAxes)
    labelLines(list(axes[row][1].get_lines()), zorder=2.5,align=True,
                   color='black',fontsize=font_size-1)
plt.savefig(f'../manuscript/figs/FigS1.pdf')
plt.show()
