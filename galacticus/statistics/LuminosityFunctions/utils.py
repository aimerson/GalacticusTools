#! /usr/bin/env python



import numpy as np
from scipy import integrate
from scipy import interpolate

from ...plotting.utils import *



def interpolateLuminosityFunction(bins,luminosityFunction,newBins,**kwargs):
    isort = np.copy(np.argsort(bins))
    bins = bins[isort]
    luminosityFunction = luminosityFunction[isort]
    f = interpolate.interp1d(bins,luminosityFunction,**kwargs)
    return f(newBins)



def integrateLuminosityFunction(bins,luminosityFunction,lowerLimit=None,upperLimit=None,**kwargs):
    # Sort tabluated luminosity function
    isort = np.copy(np.argsort(bins))
    bins = bins[isort]
    luminosityFunction = luminosityFunction[isort]
    # Set integration limits to extreme bin values if not specified
    if lowerLimit is None:
        lowerLimit = bins.min()
    if upperLimit is None:
        upperLimit = bins.max()        
    # Create function for interpolation
    kwargsInterpolate = {}
    interpolateKeys = "kind axis copy bounds_error fill_value assume_sorted".split()
    for key in interpolateKeys:
        if key in kwargs.keys():
            kwargsInterpolate[key] = kwargs[key]
    getLuminosityFunction = lambda x: interpolateLuminosityFunction(bins,luminosityFunction,x,**kwargsInterpolate)
    # Integrate interpolation function
    kwargsIntegrate = {}
    integrateKeys = "args tol rtol show divmax vec_func".split()
    for key in integrateKeys:
        if key in kwargs.keys():
            kwargsIntegrate[key] = kwargs[key]
    result = integrate.romberg(getLuminosityFunction,lowerLimit,upperLimit,**kwargsIntegrate)
    return result
