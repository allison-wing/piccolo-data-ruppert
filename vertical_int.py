#!/usr/bin/env python
# 
# Vertical integral function for calculating column water vapor (CWV) aka
# IWV aka PW.
# 
# James Ruppert  
# jruppert@ou.edu  
# March 2025

from thermo_functions import *
import numpy as np

def vert_integral_hydro(data):
    p = data['p']
    try:
        sh = data['sh']
    except:
        sh = mixr2sh(data['mr'])
    var_int = -np.trapezoid(sh, p, axis=1)/9.81
    # Replace NaNs
    nans = np.where(np.isnan(p[:,5]))
    var_int[nans] = np.nan
    return var_int

def vert_integral(data):
    p = data['p']
    t = data['tmpk']
    hght = np.repeat(data['hght'][np.newaxis,:], p.shape[0], axis=0)
    try:
        sh = data['sh']
        mr = sh2mixr(sh)
    except:
        mr = data['mr']
        sh = mixr2sh(mr)
    # rho = density_moist(t, sh, p) # kg/m3
    rd=287.04
    rv=461.5
    eps_r=rv/rd
    rho = p / ( rd * t * (1. + mr*eps_r)/(1.+mr) )
    var_int = np.trapezoid(sh * rho, hght, axis=1)
    # Replace NaNs
    nans = np.where(np.isnan(p[:,5]))
    var_int[nans] = np.nan
    return var_int
