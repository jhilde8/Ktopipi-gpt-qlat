import numpy as np

def jackblocks_all(data,omit,prt=0):
    #loop over timeslices
    #assuming shape (Ncf,...)
    jacks = np.zeros(data.shape,dtype=np.complex128)
    nconf=data.shape[0]
    other_dims=data.shape[1:]

    if(prt):print('omitting '+str(len(omit))+' of '+str(nconf)+' configs')

    #create mask for included configs
    include_mask = np.ones(nconf, dtype=bool)
    include_mask[omit] = False
    n_include = np.sum(include_mask)

    norm = 1.0 / (n_include - 1)

    total = np.sum(data[include_mask, ...], axis=0)

    jacks = np.zeros((n_include,) + other_dims, dtype=data.dtype)

    n = 0
    for i in range(nconf):
        if include_mask[i]:
            jacks[n,...] = (total - data[i,...]) * norm
            n += 1

    return jacks

#average and error calculation for a general shape vector of jackknife blocks
def jack_all(vec):
    n = vec.shape[0]
    avg = np.mean(vec,axis=0)

    err = np.zeros_like(avg)
    for i in range(n):
        err += (vec[i] - avg)*(vec[i] - avg)

    err *= (n-1)/n
    err = np.sqrt(err)

    return avg,err

#function that takes in a jackblocks array and normalizes it using the counter correlator <1>
#this function assumes the presence of the <1> correlator in the 0th expression. 
#this strips the counter index! the data is offset by 1 after this! 
def counter_norm(jk_arr):
    tol = 1e-12
    #split counters and other correlators
    counter_jks = jk_arr[:,0,...]
    corr_jks = jk_arr[:,1:,...] 
    corr_norm = np.zeros_like(corr_jks)
    nexpr = corr_jks.shape[1]

    for expr in range(nexpr):
        corr_norm[:,expr,...] = np.divide(corr_jks[:,expr,...], counter_jks, where=(counter_jks>tol))
        
    return corr_norm





