from binomial_model import *

@njit
def initialize_Hmi(mmax,epsilon):
    Hmi = np.zeros((mmax+1,mmax+1))
    for m in range(2,mmax+1):
        Hmi[m,0:m+1] = get_binom(m,epsilon)
    return Hmi

#we need to redefine this function for ame
@njit
def get_theta_bar(Hmi,thetami,pm,mvec,mmax):
    theta_bar = 0.
    norm = 0.
    for m in range(2,mmax+1):
        ivec = np.arange(m)
        theta_bar += np.sum((m-ivec)*thetami[m,0:m]*Hmi[m,0:m])*pm[m]
        norm += np.sum((m-ivec)*Hmi[m,0:m])*pm[m]
    return theta_bar/norm

@njit
def get_R(Ik, pk, kvec, kmax, theta_bar):
    R = 0.
    norm = 0.
    for k in range(1,kmax+1):
        R += k*(1-Ik[k])*pk[k]*(1-(1-theta_bar)**(k-1))
        norm += k*(1-Ik[k])*pk[k]
    return R/norm

@njit
def evolution(Hmi, Ik, pk, kvec, pm, mvec, thetami, mu):
    mmax = mvec[-1]
    kmax = kvec[-1]
    theta_bar = get_theta_bar(Hmi,thetami,pm,mvec,mmax)
    R = get_R(Ik, pk, kvec, kmax, theta_bar)
    phimi = 1-(1-thetami)*(1-R)
    Thetak = 1 - (1-theta_bar)**kvec

    #calculate new Hmi
    Hmi_new = np.zeros(Hmi.shape)
    for m in range(2,mmax+1):
        #get binomial matrices for recovery and infection
        Bmu = np.zeros((m+1,m+1))
        Bphi = np.zeros((m+1,m+1))
        for l in range(1,m+1):
            Bmu[l,0:l+1] = get_binom(l,mu)
        for l in range(m):
            Bphi[m-l,0:(m-l+1)] = get_binom(m-l,phimi[m,l])
        #apply ame
        for i in range(0,m+1):
            Hmi_new[m,i] += (1-mu)**i*Hmi[m,i]
            Hmi_new[m,i] -= (1-(1-phimi[m,i])**(m-i))*Hmi[m,i]
            if i < m:
                for j in range(1,m-i+1):
                    Hmi_new[m,i] += Bmu[i+j,j]*Hmi[m,i+j]
            if i > 0:
                for j in range(1,i+1):
                    Hmi_new[m,i] += Bphi[m-i+j,j]*Hmi[m,i-j]
    #calculate new Ik
    Ik_new = (1-mu)*Ik + (1-Ik)*Thetak
    return (Hmi_new,Ik_new)


if __name__ == '__main__':
    from scipy.special import loggamma
    from new_kernel import *
    def poisson(xvec, xmean):
        return np.exp(xvec*np.log(xmean)-xmean-loggamma(xvec+1))

    #parameter
    mu = 0.05
    f = lambda m: 1
    K = 1
    tmin = 1
    T = np.inf
    mmax = 40
    kmax = 20
    mmean = 10
    kmean = 5
    mvec = np.arange(mmax+1)
    kvec = np.arange(kmax+1)
    pm = poisson(mvec,mmean)
    pk = poisson(kvec,kmean)
    alpha = 1.5
    integrand = exponential_integrand

    beta = 0.12354903406813625
    thetami = get_thetami_mat(mmax,beta,K=K,alpha=alpha,tmin=tmin,T=T)

    epsilon = 10**(-5)
    # epsilon = 0.99
    Ik = epsilon*np.ones(kvec.shape)
    Hmi = initialize_Hmi(mmax,epsilon)
    print("----------------------")
    print("temporal evolution")
    print("----------------------")
    for t in range(100):
        # print(Hmi)
        Hmi,Ik = evolution(Hmi, Ik, pk, kvec, pm, mvec, thetami, mu)
        print(np.sum(Ik*pk))

