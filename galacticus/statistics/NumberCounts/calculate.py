#! /usr/bin/env python

import sys
import fnmatch
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romberg
from ...constants import Pi,centi,megaParsec
from ...cosmology import Cosmology,adjustHubble
from ..LuminosityFunctions.calculate import GalacticusLuminosityFunction
from ..LuminosityFunctions.utils import integrateLuminosityFunction


class ComputeNumberCounts(object):
    
    def __init__(self,luminosityFunctionObj):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store objects
        self.luminosityFunction = luminosityFunctionObj
        self.COSMOLOGY = Cosmology(omega0=self.luminosityFunction.omega0,lambda0=self.luminosityFunction.lambda0,\
                                      omegab=self.luminosityFunction.omegab,h0=self.luminosityFunction.hubble,\
                                       sigma8=self.luminosityFunction.sigma8,ns=self.luminosityFunction.ns,\
                                       h_independent=False)
        # Extract list of datasets that have luminosity function data available at all redshifts
        self.availableDatasets = None
        for out in self.luminosityFunction.outputs:            
            datasetNames = self.luminosityFunction.datasets[out]   
            if self.availableDatasets is None:
                self.availableDatasets = datasetNames
            else:
                self.availableDatasets = list(set(self.availableDatasets).intersection(datasetNames))
        return
        

    def integrateLuminosityFunctions(self,datasetName,lowerLimit,upperLimit,zmax=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")                
        zIntegral = self.luminosityFunction.redshifts
        zcut = zIntegral[zIntegral>zmax].min()
        mask = zIntegral<=zcut
        zIntegral = zIntegral[mask]
        outIntegral = np.array(self.luminosityFunction.outputs)
        lfIntegral = np.zeros_like(zIntegral)            
        # Determine luminosity function integral as function of redshift
        for iz,z in enumerate(zIntegral):
            if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):                
                radius = self.COSMOLOGY.comoving_distance(z)
                radius *= (megaParsec/centi)
                area = 4.0*Pi*radius**2
                zLowLimit = -np.log10(upperLimit*area)
                zUppLimit = -np.log10(lowerLimit*area)
                bins = 10.0**self.luminosityFunction.luminosityBins
                bins = adjustHubble(bins,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"luminosity")
                bins = -np.log10(bins)
                lf = self.luminosityFunction.getDatasets(z,required=[datasetName])[datasetName]
                lf = adjustHubble(lf,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"density")
                bins = bins[::-1]
                lf = lf[::-1]
            else:
                bcdm = self.COSMOLOGY.band_corrected_distance_modulus(z)
                zLowLimit = lowerLimit - bcdm
                zUppLimit = upperLimit - bcdm         
                bins = adjustHubble(self.luminosityFunction.magnitudeBins,\
                                     self.luminosityFunction.hubble,self.COSMOLOGY.h0,"magnitude")               
                lf = self.luminosityFunction.getDatasets(z,required=[datasetName])[datasetName]
                lf = adjustHubble(lf,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"density")            
            lfIntegral[iz] = integrateLuminosityFunction(bins,lf,lowerLimit=zLowLimit,upperLimit=zUppLimit,**kwargs)
            dA = adjustHubble(self.COSMOLOGY.angular_diameter_distance(z),\
                                  self.luminosityFunction.hubble,self.COSMOLOGY.h0,"distance")
            lfIntegral[iz] *= self.COSMOLOGY.dVdz(z)
        return zIntegral,lfIntegral


    
    def compute(self,datasetName,bins,zmin=None,zmax=None,**kwargs):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")                
        if zmin is None:
            zmin = self.luminosityFunction.redshifts.min()
        else:
            zmin = np.maximum(self.luminosityFunction.redshifts.min(),zmin)        
        if zmax is None:
            zmax = self.luminosityFunction.redshifts.max()
        else:
            zmax = np.minimum(self.luminosityFunction.redshifts.max(),zmax)        
        counts = np.zeros_like(bins)
        binWidth = bins[1] - bins[0]
        allsky = 4.0*Pi*(180.0/Pi)**2    
        kwargsInterpolate = {}
        interpolateKeys = "kind axis copy bounds_error fill_value assume_sorted".split()
        for key in interpolateKeys:
            if key in kwargs.keys():
                kwargsInterpolate[key] = kwargs[key]
        kwargsIntegrate = {}
        integrateKeys = "args tol rtol show divmax vec_func".split()
        for key in integrateKeys:
            if key in kwargs.keys():
                kwargsIntegrate[key] = kwargs[key]
        for i,bin in enumerate(bins):                       
            zIntegral,lfIntegral = self.integrateLuminosityFunctions(datasetName,bin,bin+binWidth,zmax=zmax,**kwargs)           
            fz = interp1d(np.array(zIntegral),np.array(lfIntegral),**kwargsInterpolate)
            counts[i] = romberg(fz,zmin,zmax,**kwargsIntegrate)
        counts /= (allsky*binWidth)
        return bins,counts













