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
height = width/1.8
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(width,height))

plt.subplots_adjust(left=0.2, bottom=0.12, right=0.85,
                    top=0.92, wspace=0.48, hspace=0.25)

color_dict = {0.5:"#7fcdbb", 1.:"#1d91c0", 1.5:"#0c2c84"}

#Beta distributions
# Q1 = lambda beta: beta**(-0.5)*(1-beta)**(-0.5)/np.pi
# betalim_1 = (0.,1.)
# betalim_disp_1 = (0.01,0.99)
Q1 = lambda beta: 8**2*beta*np.exp(-8*beta)
betalim_1 = (0.,np.inf)
betalim_disp_1 = (0.,1.5)
Q2 = lambda beta: 1.2*(0.1)**(1.2)/beta**(2.2)
betalim_2 = (0.1,np.inf)
betalim_disp_2 = (0.1,1.5)
Qlist = [Q1,Q2]
betalimlist = [betalim_1,betalim_2]
betalimdisplist = [betalim_disp_1,betalim_disp_2]


for ind, (Q,betalim,betalim_disp) in enumerate(zip(Qlist,betalimlist,
                                                   betalimdisplist)):
    #left plot : beta distribution
    #-------------------
    # Q = lambda beta: beta**(-0.5)*(1-beta)**(-0.5)/np.pi
    betavec = np.linspace(betalim_disp[0],betalim_disp[1],1000)
    Qvec = Q(betavec)
    axes[ind][0].plot(betavec,Qvec, color="#41b6c4")
    axes[ind][0].fill_between(betavec,Qvec, color="#41b6c4", alpha=0.3)
    if ind == 1:
        axes[ind][0].set_xlabel(r"$\beta$")
    axes[ind][0].set_ylabel(r"$Q(\beta)$")
    if ind == 0:
        axes[ind][0].text(0., 1.1, r'(a) \, Heterogeneity',
                 fontsize=font_size+1,
                 transform=axes[ind][0].transAxes)

    #right plot : kernel
    #-------------------
    alpha_list = [0.5,1.,1.5]
    rho_list = np.logspace(-2,0,20)
    T = np.inf

    compute = False

    if compute:
        kernel_list = []
        for alpha in alpha_list:
            kernel = [kernel_het_beta(rho,alpha=alpha,Q=Q,betalim=betalim)
                      for rho in rho_list]
            kernel_list.append(kernel)
        with open(f'./dat/kernel_het_beta_{ind}.pk', 'wb') as filename:
            pickle.dump(kernel_list,filename)
    else:
        with open(f'./dat/kernel_het_beta_{ind}.pk', 'rb') as filename:
            kernel_list = pickle.load(filename)

    # if ind != 1:
        # axes[ind][1].text(0.5, 0.15, r'$\theta_m(\rho) \propto \rho^\alpha$',
                 # fontsize=font_size+1,
                 # transform=axes[ind][1].transAxes)

        # exponent = "{1/2}"
        # label=fr"$\rho^{exponent}$"
        # axes[ind][1].loglog(rho_list,1.2*rho_list**(alpha_list[0]-1), ':',
                            # color="#1a1a1a",
                      # label=label)
        # exponent = "{1/5}"
        # label=fr"$\rho^{exponent}$"
        # axes[ind][1].loglog(rho_list,1.*rho_list**(1/5), ':',
                            # color="#1a1a1a",
                      # label=label)
    for alpha,kernel in zip(alpha_list,kernel_list):
        label=fr"$\alpha = {alpha}$" #change notation alpha -> alpha - 1
        axes[ind][1].loglog(rho_list,kernel, '-', color=color_dict[alpha],
                      label=label)

    if ind == 1:
        # axes[ind][1].text(0.5, 0.15, r'$\theta_m(\rho) \propto \rho^{1/5}$',
                 # fontsize=font_size+1,
                 # transform=axes[ind][1].transAxes)
        axes[ind][1].loglog(rho_list,0.1*rho_list**1.2,'--',color='#1a1a1a',
                            label=r"$\propto \rho^\psi$")

    # if ind != 2:
        # exponent = "{3/2}"
        # label=fr"$\rho^{exponent}$"
        # axes[ind][1].loglog(rho_list,0.003*rho_list**(alpha_list[-1]-1),
                       # '--', color="#1a1a1a", label=label)
    labelLines(list(axes[ind][1].get_lines()), zorder=2.5,align=True,
               color='black',fontsize=font_size-1)
    if ind == 1:
        axes[ind][1].set_xlabel(r"Environment prevalence $\rho$")
    axes[ind][1].set_ylabel(r"Infection probability $\theta_m$")
    if ind == 0:
        axes[ind][1].text(0., 1.1, r'(b) \, Infection kernel',
                          fontsize=font_size+1,
                 transform=axes[ind][1].transAxes)

plt.savefig('../manuscript/figs/FigS4.pdf')
# plt.show()
