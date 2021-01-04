from astropy.io import ascii
from astropy.io import fits
import numpy as np
from scipy.integrate import simps
from scipy.integrate import cumtrapz
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

def cal_flux(flux,lamda,lamda_all,var,T):
    # Calculate OZDES flux convolving with DES transmission function
    # flux/var/lamda_all is the raw spectra from OZDES
    # T/lamda is from DES transmission curve   
    c = 2.992792e18
    tmp_num = np.nansum(np.array(flux)*np.array(T)*lamda)
    #tmp_num = simps(np.array(flux)*np.array(T)*lamda,lamda)
    #print(flux)
    if tmp_num < 0:
        index = np.where((flux > 0), True, False)
        #print(flux)
        flux = np.interp(lamda,flux[index],lamda[index])
        tmp_num = np.nansum(flux*T*lamda)
        
    tmp_den = c * np.nansum(np.array(T)/lamda)
    #tmp_den = c * simps(np.array(T)/lamda,lamda)
    tmp = -2.5 * np.log10(tmp_num / tmp_den) - 48.60
    
    var = np.interp(lamda,lamda_all,var)
    num_var = np.nansum(var * (flux * lamda) ** 2)
    tmp_var = 1.17882 * num_var / (tmp_num ** 2)
    
    #print(tmp_num)
    return tmp,tmp_var

def cal_ratio(a,b,a_err,b_err):
    # Calculate scale ratio for DES gri bands by comparing
    # the difference of OZDES instrumental luminosity and DES luminosity
    # a/a_err and b/b_err are DES and OZDES luminosity respectively
    scale = np.power(10., 0.4 * (a - b))
    scale_err = abs(b_err + a_err ) * (scale * 0.4 * 2.3) ** 2
    return scale,scale_err

def cal_scale_factor(scale,scale_err,lamda,lamdas,tmp_var):
    # Calculate the scale factor for all wavelength
    # scale/scale_err is the output from cal_ratio
    # lamdas is the effective wavelength for DES bands
    # tmp_var is the output from cal_flux
    scale = np.array(scale)
    scale_err = np.array(scale_err)
    # using 2-d polynomials to construct scale factor for OZDES spectra
    fn_raw = np.polyfit(lamdas,scale,2)
    fn = np.poly1d(fn_raw)
    flux_calib = fn(lamda)
    
    # Calculate the variance for calibrated spectra
    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scale_err ** 0.5) / 10 ** -17
    scale_v = scale / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(.01, 2000.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev**2)

    xprime = np.atleast_2d(lamdas).T
    yprime = np.atleast_2d(scale_v).T
    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(lamda).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)
    y_pred = y_pred[:,0]
    sigModel = (sigma/y_pred)*flux_calib

    # now scale the original variance and combine with scale factor uncertainty
    varScale = tmp_var * pow(flux_calib, 2) + sigModel ** 2
    return flux_calib, varScale