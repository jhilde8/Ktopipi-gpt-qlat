import numpy as np

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