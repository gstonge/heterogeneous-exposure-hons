from scipy.integrate import quad,dblquad
import numpy as np
from scipy.special import gamma
from scipy.special import gammainc

def poisson_integrand(tau, rho, beta, fm=1, K=1, alpha=2):
    #lambda = beta*f(m)*rho*tau
    L = np.array([(tau*beta*rho*fm)**k/gamma(k+1) for k in range(K)])
    return (1-np.exp(-tau*beta*rho*fm)*np.sum(L))*tau**(-alpha-1)

def exponential_integrand(tau, rho, beta, fm=1, K=1, alpha=2):
    scale = tau*rho*beta*fm
    return np.exp(-K/scale)*tau**(-alpha-1)

def exponential_pdf(kappa,scale=1):
    return np.exp(-kappa/scale)/scale

def weibull_integrand(tau, rho, beta, fm=1, K=1, alpha=2, shape=2):
    scale = tau*rho*beta*fm/gamma(1+1/shape) #mean = tau*rho*beta
    return np.exp(-(K/scale)**shape)*tau**(-alpha-1)

def weibull_pdf(kappa,scale=1,shape=2):
    return (shape/scale)*(kappa/scale)**(shape-1)*np.exp(-(kappa/scale)**shape)

def frechet_integrand(tau, rho, beta, fm=1, K=1, alpha=2, shape=2):
    scale = tau*rho*beta*fm/gamma(1-1/shape)
    return (1-np.exp(-(K/scale)**(-shape)))*tau**(-alpha-1)

def frechet_pdf(kappa,scale=1,shape=2):
    return (shape/scale)*(kappa/scale)**(-shape-1)*np.exp(-(kappa/scale)**(-shape))

def gamma_special_integrand(tau, rho, beta, fm=1, K=1, alpha=2 ,z = 0.):
    #play with the scale/shape instead of just the scale, such that variance != mean^2
    #z = 0 is equivalent to the exponential
    param = tau*rho*beta*fm
    return (1 - gammainc(param**z,K/param**(1-z)))*tau**(-alpha-1)

def kernel(rho, beta, fm=1, K=1, alpha=2., tmin=1, T=np.inf,
           integrand=exponential_integrand,
           args=tuple()):
    Z = (tmin**(-alpha)-T**(-alpha))/alpha
    _args = (rho,beta,fm,K,alpha,*args)
    return quad(integrand,tmin,T,args=_args)[0]/Z

#same as kernel, but put beta first for integration and multiply by Q(beta)
def kernel2(beta, Q, rho, fm=1, K=1, alpha=2.,tmin=1, T=np.inf,
           integrand=exponential_integrand,
           args=tuple()):
    _args = (rho,beta,fm,K,alpha, *args)
    return Q(beta)*quad(integrand,tmin,T,args=_args)[0]


def kernel_het_beta(rho, fm=1, K=1, alpha=2., tmin=1, T=np.inf,
                    integrand=exponential_integrand,args=tuple(),
                    Q=lambda b: np.exp(-b),betalim=(0,np.inf)):
    Z = (tmin**(-alpha)-T**(-alpha))/alpha
    _args=(Q,rho,fm,K,alpha,tmin,T,integrand,args)
    return quad(kernel2,betalim[0],betalim[1],args=_args)[0]/Z

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    alpha_list = [0.5,1.,1.5,2.]
    rho_list = np.logspace(-3,0,100)
    beta = 0.1
    for alpha in alpha_list:
        label=fr"$\alpha = {alpha}$"
        kernel_list = [kernel(rho,alpha,beta,K=0.1,tmin=1,
                              integrand=gamma_integrand,
                              args=tuple())
                       for rho in rho_list]
        plt.loglog(rho_list,kernel_list, '-',label=label)
        plt.loglog(rho_list,rho_list**alpha, '--',label=label)

    plt.legend()
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\theta_m(\rho)$")
    plt.show()
