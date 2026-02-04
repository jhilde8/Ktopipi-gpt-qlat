import numpy as np
from scipy.optimize import curve_fit
from .jackknife import *

#calculate covariance matrix with shrinkage
def covariance(data, shrinkage=0.1):

    nconf,tsize=data.shape
    covmat=np.zeros((tsize,tsize))
    avg = np.average(data,axis=0)
    for n in range(nconf):
        covmat+=np.outer(data[n]-avg,data[n]-avg)
    covmat*=1/nconf
    covmat *= nconf-1

    #shrink toward diagonal
    diag_cov = np.diag(np.diag(covmat))
    covmat_shrunk = (1-shrinkage) * covmat + shrinkage * diag_cov

    return avg,covmat_shrunk


#fit models
#calcualtes the variance of a multi-paramter function, where c is a square matrix with dimensions of the same size as the number of parameters.
def err_const_plus_exp(t,p,c):

    fact = np.exp(-t*p[2])
    errsq  =   c[0][0]
    errsq +=   c[1][1]*(fact)**2
    errsq +=   c[2][2]*((-t)*p[1]*fact)**2
    errsq += 2*c[0][1]*fact
    errsq += 2*c[0][2]*(-t)*p[1]*fact
    errsq += 2*c[1][2]*(-t)*p[1]*fact**2

    return np.sqrt(errsq)

def const_fit(t,*p):

    return p[0]

def pure_exp(t,*p):
    return p[0]*np.exp(-t*p[1])

def two_exp(t, A0, E0):
    NT = 32 #adjust.
    return (A0*np.exp(-E0*t) + A0*np.exp(-(NT-t)*E0))

def two_state_two_exp(t, A0, E0, A1, E1):
    NT = 32
    model = (A0*np.exp(-E0*t) + A0*np.exp(-(NT-t)*E0)) + (A1*np.exp(-E1*t) + A1*np.exp(-(NT-t)*E1))
    return model
    

#single trajectory fit
def c2pt_fit_one_traj(data, tmin, tmax):
    t_fit = np.arange(tmin,tmax)

    data_range = data[tmin:tmax]
    p0 = [data_range[0],0.1]

    popt, pcov = curve_fit(pion_model_single, t_fit, data_range, p0, maxfev=10000)

    A0, E0 = popt
    E0_err = np.sqrt(pcov[1,1])

    print(f"Single trajectory exponential fit: E0: {E0} +/- {E0_err}")

#2pt function fitting function
def c2pt_1state_fit(data_jk, fit_range, fit_model, shrinkage):
    tmin, tmax = fit_range
    ncf = data_jk.shape[0]
    t_fit = np.arange(tmin,tmax+1)

    #corr_avg, corr_err = jack_all(data_jk) uncorrelated fit

    corr_avg, covmat = covariance(data_jk[:, tmin:tmax+1], shrinkage)
    cond = np.linalg.cond(covmat)
    print(f"Covariance matrix condition number: {cond}")

    data_range = corr_avg
    E_guess = np.log(corr_avg[0]/corr_avg[1])
    p0 = [data_range[0],E_guess]
    
    #calculate covariance matrix
    #fit of the average correlator using the average variance as sigma
    popt_avg, pcov_avg = curve_fit(fit_model,
                                  t_fit,
                                  corr_avg,
                                  #sigma=corr_err[tmin:tmax+1], uncorrelated fit
                                  sigma=covmat,
                                  absolute_sigma=True,
                                  p0=p0,
                                  maxfev=50000)

    param_names = ['A0', 'E0']
    fit_params = {key: [] for key in param_names}
    params_jk = np.zeros((ncf,2)) #array to hold params for eahc jackknife block

    #fit of each jackknife block, no weights to generate errors
    for cf in range(ncf):
        y_data = data_jk[cf, tmin:tmax+1]
        popt, _ = curve_fit(fit_model,
                           t_fit,
                           y_data,
                           p0=popt_avg,
                           maxfev=50000)
        for key, value in zip(param_names, popt):
            fit_params[key].append(value)

        params_jk[cf] = popt

    fit_errors = {}
    for key in param_names:
        vals = np.array(fit_params[key])
        mean_val, err_val = jack_all(vals)
        fit_errors[key] = err_val

    #calculate chi squared
    corr_model = fit_model(t_fit, *popt_avg)
    residuals = corr_avg - corr_model
    chi2 = np.sum((residuals / np.std(residuals))**2)
    dof = len(t_fit) - 2
    chi2_dof = chi2/dof

    #covariance matrix
    params_cov = np.cov(params_jk.T) * (ncf - 1)

    return popt_avg,fit_errors,fit_params,chi2_dof

def c2pt_2state_fit(data_jk, fit_range, shrinkage):
    tmin, tmax = fit_range
    ncf = data_jk.shape[0]
    t_fit = np.arange(tmin,tmax+1)

    #corr_avg, corr_err = jack_all(data_jk)
    corr_avg, covmat = covariance(data_jk[:,tmin:tmax+1], shrinkage)
    cond = np.linalg.cond(covmat)
    print(f"Covariance matrix condition number: {cond}")
    
    data_range = corr_avg
    E_guess = np.log(corr_avg[0]/corr_avg[1])
    print(E_guess)
    
    p0 = [data_range[0],E_guess, data_range[0], E_guess*2]
    
    #calculate covariance matrix
    #fit of the average correlator using the average variance as sigma
    popt_avg, pcov_avg = curve_fit(two_state_two_exp,
                                  t_fit,
                                  corr_avg,
                                  sigma=covmat,
                                  absolute_sigma=True,
                                  p0=p0,
                                  maxfev=150000)

    param_names = ['A0', 'E0', 'A1', 'E1']
    fit_params = {key: [] for key in param_names}
    params_jk = np.zeros((ncf,4)) #array to hold params for eahc jackknife block

    #fit of each jackknife block, no weights to generate errors
    for cf in range(ncf):
        y_data = data_jk[cf, tmin:tmax+1]
        popt, _ = curve_fit(two_state_two_exp,
                           t_fit,
                           y_data,
                           p0=popt_avg,
                           maxfev=100000)
        for key, value in zip(param_names, popt):
            fit_params[key].append(value)

        params_jk[cf] = popt

    fit_errors = {}
    for key in param_names:
        vals = np.array(fit_params[key])
        mean_val, err_val = jack_all(vals)
        fit_errors[key] = err_val

    #calculate chi squared
    corr_model = two_state_two_exp(t_fit, *popt_avg)
    residuals = corr_avg - corr_model
    chi2 = np.sum((residuals / np.std(residuals))**2)
    dof = len(t_fit) - 2
    chi2_dof = chi2/dof

    #covariance matrix
    params_cov = np.cov(params_jk.T) * (ncf - 1)

    return popt_avg,fit_errors,fit_params,chi2_dof

def meson_energy_1state(data_jk, tmin, tmax, fit_model=None, shrinkage=1.0):
    if fit_model == None:
        fit_model = pure_exp
    
    fit_range = (tmin, tmax)
    popt_avg, fit_errors, fit_params, chi2_dof = c2pt_1state_fit(data_jk, fit_range, fit_model, shrinkage)
    E0 = popt_avg[1]
    E0_err = fit_errors['E0']
    E0_block = fit_params['E0']
    A0 = popt_avg[0]
    A0_err = fit_errors['A0']
    A0_block = fit_params['A0']

    #print(f"Fit energy: {E0} +/- {E0_err}")

    return E0, E0_err

def meson_energy_2state(data_jk, tmin, tmax, shrinkage):
    fit_range = (tmin, tmax)
    popt_avg, fit_errors, fit_params, chi2_dof = c2pt_2state_fit(data_jk, fit_range, shrinkage)
    E0 = popt_avg[1]
    E0_err = fit_errors['E0']
    E0_block = fit_params['E0']
    A0 = popt_avg[0]
    A0_err = fit_errors['A0']
    A0_block = fit_params['A0']

    E1 = popt_avg[3]
    E1_err = fit_errors['E1']
    E1_block = fit_params['E1']
    A1 = popt_avg[2]
    A0_err = fit_errors['A1']
    A1_block = fit_params['A1']
    
    return E0, E0_err, E1, E1_err

#effective mass fitting function
def eff_mass_fit(data_jk, fit_range):
    tmin, tmax = fit_range
    ncf = data_jk.shape[0]
    t_fit = np.arange(tmin,tmax+1)

    corr_avg, corr_err = jack_all(data_jk)

    #calculate covariance matrix
    #fit of the average correlator using the average variance as sigma
    popt_avg, pcov_avg = curve_fit(constant_model,
                                  t_fit,
                                  corr_avg[tmin:tmax+1],
                                  sigma=corr_err[tmin:tmax+1],
                                  p0=[0.2],
                                  maxfev=50000)

    param_names = ['E0']
    fit_params = {key: [] for key in param_names}
    params_jk = np.zeros((ncf)) #array to hold params for each jackknife block

    #fit of each jackknife block, no weights to generate errors
    for cf in range(ncf):
        y_data = data_jk[cf, tmin:tmax+1]
        popt, _ = curve_fit(constant_model,
                           t_fit,
                           y_data,
                           p0=popt_avg,
                           maxfev=50000)

        fit_params['E0'].append(popt[0])
        params_jk[cf] = popt[0]


    E0_vals = np.array(fit_params['E0'])
    E0_mean, E0_err = jack_all(E0_vals)
    fit_errors = {'E0': E0_err}

    #calculate chi squared
    corr_model = constant_model(t_fit, *popt_avg)
    residuals = corr_avg[tmin:tmax+1] - corr_model
    chi2 = np.sum((residuals / corr_err[tmin:tmax+1])**2)
    dof = len(t_fit) - 1
    chi2_dof = chi2/dof

    #covariance matrix
    params_cov = np.cov(params_jk) * (ncf - 1)

    return popt_avg,fit_errors,fit_params,chi2_dof

def eff_mass_energy(data_jk, tmin, tmax):
    fit_range = (tmin, tmax)
    popt_avg, fit_errors, fit_params, chi2_dof = eff_mass_fit(data_jk, fit_range)
    E0 = popt_avg[0]
    E0_err = fit_errors['E0']
    E0_block = fit_params['E0']
    return E0, E0_err, E0_block, chi2_dof


