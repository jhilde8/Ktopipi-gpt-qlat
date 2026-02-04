import numpy as np

def emp_fold(data):
    ncf = data.shape[0]
    nt = data.shape[1]
    E_data = np.zeros((ncf,(nt//2)-3))

    print(nt//2 - 1)
    t = 0
    for cf in range(ncf):
        C = data[cf,:]
        CF = [((C[t] + C[(nt - t) % nt]) / 2) for t in range(nt//2 - 1)]
        E = [np.arccosh((CF[t] + CF[t+2])/2/CF[t+1]) for t in range(0,nt//2 - 3)]

        E_data[cf,:] = E[:]

    return E_data

#effective mass calculation using two data points/log definition. This uses only one half of the correlator data. This is for the one term fit function
def eff_mass_2d_ln_half(corr_data):
    ncf = corr_data.shape[0]
    nt = corr_data.shape[1]
    m_data = np.zeros((ncf, nt-2),np.float64)
    E = 0.0
    for cf in range(ncf):
        for t in range(nt-2):
        #E = np.arccosh((C_fold[t] + C_fold[t+2])/(2*C_fold[t+1]))
            E = np.log(corr_data[cf,t]/corr_data[cf,t+1])
            m_data[cf,t] = E

    return m_data

#effective mass calculation using two data points/log definition. 
def eff_mass_2d_ln(corr_data):
    ncf = corr_data.shape[0]
    nt = corr_data.shape[1]
    m_data = np.zeros((ncf, nt//2-2),np.float64)
    E = 0.0
    for cf in range(ncf):
        #fold correlator
        C_fold = np.zeros(nt//2 + 1,np.float64)

        C_fold[0] = corr_data[cf,0]
        C_fold[nt//2] = corr_data[cf,nt//2]
        for t in range(1,nt//2):
            C_fold[t] = (corr_data[cf,t] + corr_data[cf,nt-t])/2 
        
        for t in range(nt//2-2):
        #E = np.arccosh((C_fold[t] + C_fold[t+2])/(2*C_fold[t+1]))
            #E = np.log(corr_data[cf,t]/corr_data[cf,t+1])
            E = np.log(C_fold[t]/C_fold[t+1])
            m_data[cf,t] = E

    return m_data
        

def eff_mass_2d_cosh_half(corr_data):
    ncf = corr_data.shape[0]
    nt = corr_data.shape[1]
    m_data = np.zeros((ncf, nt-2),np.float64)
    E = 0.0
    for cf in range(ncf):
        for t in range(nt-2):
            E = np.arccosh((corr_data[cf,t] + corr_data[cf,t+2])/(2*corr_data[cf,t+1]))
            print(E)
            m_data[cf,t] = E

    return m_data

#Effective mass calculation using three data points and the arccosh definition. This folds the correlator, and uses two data points per measurement
#this is to be used with the two term fit function
def eff_mass_2d_cosh(corr_data):
    ncf = corr_data.shape[0]
    nt = corr_data.shape[1]
    m_data = np.zeros((ncf, nt//2-2),np.float64)
    E = 0.0
    for cf in range(ncf):
        #fold correlator
        C_fold = np.zeros(nt//2 + 1,np.float64)

        C_fold[0] = corr_data[cf,0]
        C_fold[nt//2] = corr_data[cf,nt//2]
        for t in range(1,nt//2):
            C_fold[t] = (corr_data[cf,t] + corr_data[cf,nt-t])/2 
        
        for t in range(nt//2-2):
            E = np.arccosh((C_fold[t] + C_fold[t+2])/(2*C_fold[t+1]))
            m_data[cf,t] = E

    return m_data


def GEVP(ens_params,minOp,maxOp,tmin,tmax):
    ncf = ens_params.ncf
    nt = ens_params.nt
    ens_params.e = np.zeros((ncf, nt, maxOp-minOp))
    ens_params.vec=np.zeros((ncf, nt, maxOp-minOp, maxOp-minOp))
    
    for cf in range(ncf):
        for t in range(tmin,tmax):
            a = ens_params.jks[minOp:maxOp, minOp:maxOp, cf, t] #C(t) for some jackknife block
            b = ens_params.jks[minOp:maxOp, minOp:maxOp, cf, t-1] #C(t_0) for some jackknife block
            #np.linalg.cholesky(b) error here if b is not positive definite
            a = 0.5*(a+np.conjugate(np.transpose(a))) #ensure hermiticity
            b = 0.5*(b+np.conjugate(np.transpose(b)))

            ev, evec = eigh(a,b,type=1)

            #sort the eigenvalues into descending order
            ev_desc = ev[::-1]
            ens_params.e[cf,t] = ev_desc
            #sort the columns of the eigenvectors to match
            vec_desc = evec[:,::-1]
            ens_params.vec[cf,t] = vec_desc
            #ens_params.e[cf,t],ens_params.vec[cf,t]=eigh(a,b,type=1) # Type 1 => a @ v = w @ b @ v


    