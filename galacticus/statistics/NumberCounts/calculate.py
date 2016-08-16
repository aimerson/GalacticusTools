#! /usr/bin/env python

import sys
import fnmatch
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romberg
from ...utils.progress import Progress
from ...constants import Pi,centi,megaParsec
from ...cosmology import Cosmology,adjustHubble
from ..LuminosityFunctions.calculate import GalacticusLuminosityFunction
from ..LuminosityFunctions.utils import integrateLuminosityFunction









class NumberCounts(object):
    
    def __init__(self,luminosityFunctionObj,hubble=None):               
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store objects
        self.luminosityFunction = luminosityFunctionObj
        if hubble is None:
            hubble = self.luminosityFunction.hubble
        self.COSMOLOGY = Cosmology(omega0=self.luminosityFunction.omega0,lambda0=self.luminosityFunction.lambda0,\
                                      omegab=self.luminosityFunction.omegab,h0=hubble,\
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



    def getLuminosityLimits(self,redshift,faintFluxLimit,brightFluxLimit=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        radius = self.COSMOLOGY.comoving_distance(redshift)
        radius *= (megaParsec/centi)
        area = 4.0*Pi*(radius**2)                        
        faintFluxLimit = 10.0**faintFluxLimit
        faintLuminosityLimit = np.log10(faintFluxLimit*area)
        if brightFluxLimit is None:
            brightLuminosityLimit = None
        else:            
            brightFluxLimit = 10.0**brightFluxLimit
            brightLuminosityLimit = np.log10(brightFluxLimit*area)
        return faintLuminosityLimit,brightLuminosityLimit


    def getAbsMagnitudeLimits(self,redshift,faintAppLimit,brightAppLimit=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        bcdm = self.COSMOLOGY.band_corrected_distance_modulus(redshift)
        faintAbsLimit = faintAppLimit - bcdm           
        if brightAppLimit is None:
                    brightAbsLimit = None
        else:
            brightAbsLimit = brightAppLimit - bcdm           
        return faintAbsLimit,brightAbsLimit

    def _get_interpolate_keywords(self,**kwargs):
        kwargsInterpolate = {}
        interpolateKeys = "kind axis copy bounds_error fill_value assume_sorted".split()
        for key in interpolateKeys:
            if key in kwargs.keys():
                kwargsInterpolate[key] = kwargs[key]
        return kwargsInterpolate

    def _get_integrate_keywords(self,**kwargs):
        kwargsIntegrate = {}
        integrateKeys = "args tol rtol show divmax vec_func".split()
        for key in integrateKeys:
            if key in kwargs.keys():
                kwargsIntegrate[key] = kwargs[key]
        return kwargsIntegrate


    def integrateLuminosityFunctions(self,datasetName,faintLimit,brightLimit=None,zmin=None,zmax=None,verbose=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")                
        zIntegral = self.luminosityFunction.redshifts
        # Select only redshift snapshots spanning redshift range of interest
        if zmax is not None:            
            if zmax < zIntegral.max():
                zcut = zIntegral[zIntegral>zmax].min()
                mask = zIntegral<=zcut
                zIntegral = zIntegral[mask]
        if zmin is not None:
            if zmin > zIntegral.min():
                zcut = zIntegral[zIntegral<zmin].max()
                mask = zIntegral>=zcut
                zIntegral = zIntegral[mask]
        lfIntegral = np.zeros_like(zIntegral)                    
        # Determine luminosity function integral as function of redshift
        if verbose:
            print(funcname+"(): Computing luminosity function for Galacticus snapshot outputs...")
        PROG = Progress(len(zIntegral))
        for iz,redshift in enumerate(zIntegral):
            if redshift == 0.0:
                z = 1.0e-3
            else:
                z = redshift
            if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):   
                zfaintLimit,zbrightLimit = self.getLuminosityLimits(z,faintLimit,brightFluxLimit=brightLimit)                            
                bw = self.luminosityFunction.luminosityBins[1] - self.luminosityFunction.luminosityBins[0]
                bins = 10.0**self.luminosityFunction.luminosityBins
                bins = adjustHubble(bins,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"luminosity")
                lf = self.luminosityFunction.getDatasets(z,required=[datasetName])[datasetName]            
                lf *= bw
                lf = adjustHubble(lf,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"density")
                bins = np.log10(bins)
                lfIntegral[iz] = integrateLuminosityFunction(bins,lf,lowerLimit=zfaintLimit,upperLimit=zbrightLimit,**kwargs)
            else:
                zfaintLimit,zbrightLimit = self.getAbsMagnitudeLimits(z,faintLimit,brightAppLimit=brightLimit)
                bins = adjustHubble(self.luminosityFunction.magnitudeBins,\
                                     self.luminosityFunction.hubble,self.COSMOLOGY.h0,"magnitude")               
                lf = self.luminosityFunction.getDatasets(z,required=[datasetName])[datasetName]
                lf *= bw
                lf = adjustHubble(lf,self.luminosityFunction.hubble,self.COSMOLOGY.h0,"density")       
                lfIntegral[iz] = integrateLuminosityFunction(bins,lf,lowerLimit=zbrightLimit,upperLimit=zfaintLimit,**kwargs)
            lfIntegral[iz] *= self.COSMOLOGY.dVdz(z)
            PROG.increment()
            if verbose:
                PROG.print_status_line()
        return zIntegral,lfIntegral




class fluxNumberCounts(NumberCounts):
    
    def __init__(self,luminosityFunctionObj,hubble=None):
        super(fluxNumberCounts,self).__init__(luminosityFunctionObj,hubble=hubble)
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return
    
    def computeCounts(self,datasetName,bins,zmin=None,zmax=None,cumulative=False,verbose=False,**kwargs):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset available in luminosity function file
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")                
        # Construct array to store counts
        counts = np.zeros_like(bins)
        # Set redshift limits
        if zmin is None:
            zmin = self.luminosityFunction.redshifts.min()
        else:
            zmin = np.maximum(self.luminosityFunction.redshifts.min(),zmin)        
        if zmax is None:
            zmax = self.luminosityFunction.redshifts.max()
        else:
            zmax = np.minimum(self.luminosityFunction.redshifts.max(),zmax)            
        # Compute counts
        if verbose:
            print(funcname+"(): Computing number counts in each flux/magnitude bin for dataset '"+datasetName+"'...")
        kwargsInterpolate = self._get_interpolate_keywords(**kwargs)
        kwargsIntegrate = self._get_integrate_keywords(**kwargs)
        binWidth = bins[1] - bins[0]
        PROG = Progress(len(bins))        
        for i,bin in enumerate(bins):                       
            if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):
                lowLimit = bin
                uppLimit = bin + binWidth
            else:
                lowLimit = bin + binWidth
                uppLimit = bin
            if cumulative:
                uppLimit = None            
            zIntegral,lfIntegral = self.integrateLuminosityFunctions(datasetName,lowLimit,brightLimit=uppLimit,zmin=zmin,zmax=zmax,\
                                                                         cumulative=cumulative,verbose=False,**kwargs)                           
            fz = interp1d(np.array(zIntegral),np.array(lfIntegral),**kwargsInterpolate)
            counts[i] = romberg(fz,zmin,zmax,**kwargsIntegrate)
            PROG.increment()
            if verbose:
                PROG.print_status_line()
        allsky = 4.0*Pi*(180.0/Pi)**2    
        counts /= allsky
        if not cumulative:
            counts /= binWidth
        return bins,counts




class redshiftNumberCounts(NumberCounts):
    
    def __init__(self,luminosityFunctionObj):
        super(redshiftNumberCounts,self).__init__(luminosityFunctionObj)
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self._datasetName = None
        self._zIintegral = None
        self._phiIntegral = None
        return
    

    
    def computeCounts(self,datasetName,bins,faintLimit,brightLimit=None,verbose=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")

        zIntegral,phiIntegral = self.integrateLuminosityFunctions(datasetName,\
                                                                      faintLimit,brightLimit=brightLimit,\
                                                                      zmax=bins.max(),cumulative=False,\
                                                                      verbose=False,**kwargs)                        
        kwargsInterpolate = self._get_interpolate_keywords(**kwargs)    
        binWidth = bins[1] - bins[0]
        allsky = 4.0*Pi*(180.0/Pi)**2
        fz = interp1d(np.array(zIntegral),np.array(phiIntegral),**kwargsInterpolate)
        counts = fz(bins)
        counts /= (allsky*binWidth)
        return bins,counts




