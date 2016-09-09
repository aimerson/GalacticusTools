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
from ...hdf5 import HDF5



class NumberCounts(object):
    
    def __init__(self,luminosityFunctionObj,hubble=None):               
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store objects
        self.luminosityFunction = luminosityFunctionObj
        # Extract list of datasets that have luminosity function data available at all redshifts
        self.availableDatasets = self.luminosityFunction.availableDatasets
        # Store list of available redshifts
        self.redshifts = np.sort(np.array([self.luminosityFunction.outputs[k] \
                                               for k in self.luminosityFunction.outputs.keys()]))
        self.redshifts = np.sort(self.redshifts).astype(float)
        # Create empty array to store redshift integral
        self.redshiftIntegral = np.zeros_like(self.redshifts)
        # Store cosmology
        if hubble is None:
            self.hubble = self.luminosityFunction.cosmology["HubbleParameter"]
        else:
            self.hubble = hubble
        self.COSMOLOGY = Cosmology(omega0=self.luminosityFunction.cosmology["OmegaMatter"],\
                                       lambda0=self.luminosityFunction.cosmology["OmegaDarkEnergy"],\
                                       omegab=self.luminosityFunction.cosmology["OmegaBaryon"],\
                                       h0=self.luminosityFunction.cosmology["HubbleParameter"],\
                                       sigma8=self.luminosityFunction.cosmology["sigma8"],\
                                       ns=self.luminosityFunction.cosmology["ns"],\
                                       h_independent=False,zmax=self.redshifts.max()*1.1)
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


    def _integrateLuminosityFunctionAtRedshift(self,iz,datasetName,faintLimit,brightLimit=None,resolutionLimit=None,\
                                                   cumulative=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        # Get redshift (avoid z=0 exactly)
        redshift = self.redshifts[iz]
        if redshift == 0.0:
            redshift = 1.0e-3
        # Get counts at this redshift
        hubbleLF = self.luminosityFunction.cosmology["HubbleParameter"]
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):   
            zLowLimit,zUppLimit = self.getLuminosityLimits(redshift,faintLimit,brightFluxLimit=brightLimit)                            
            bins = 10.0**self.luminosityFunction.luminosityBins
            bins = adjustHubble(bins,hubbleLF,self.hubble,"luminosity")
            bins = np.log10(bins)
            if resolutionLimit is not None:
                mask = bins > resolutionLimit
            else:
                mask = np.ones(len(bins)).astype(bool)
        else:
            zUppLimit,zLowLimit = self.getAbsMagnitudeLimits(redshift,faintLimit,brightAppLimit=brightLimit)
            bins = adjustHubble(self.luminosityFunction.magnitudeBins,hubbleLF,self.hubble,"magnitude")               
            if resolutionLimit is not None:
                mask = bins < resolutionLimit
            else:
                mask = np.ones(len(bins)).astype(bool)
        lf = self.luminosityFunction.getDatasets(redshift,required=[datasetName])[datasetName]
        lf = lf[mask]
        bins = bins[mask]       
        bw = bins[1] - bins[0]
        if cumulative:
            lf *= bw
        lf = adjustHubble(lf,hubbleLF,self.hubble,"density")       
        self.redshiftIntegral[iz] = integrateLuminosityFunction(bins,lf,lowerLimit=zLowLimit,\
                                                                    upperLimit=zUppLimit,**kwargs)
        self.redshiftIntegral[iz] *= self.COSMOLOGY.dVdz(redshift)        
        return


    def integrateLuminosityFunctions(self,datasetName,faintLimit,brightLimit=None,zmin=None,zmax=None,cumulative=False,\
                                         verbose=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name 
        if datasetName not in self.availableDatasets:
            raise ValueError(funcname+"(): '"+datasetName+"' not in list of available datasets!")                
        # Select only redshift snapshots spanning redshift range of interest
        izUpp = np.arange(len(self.redshifts)).max()
        if zmax is not None:            
            if zmax < self.redshifts.max():
                izUpp = np.argwhere(self.redshifts>zmax).min()
        izLow = 0
        if zmin is not None:            
            if zmin > self.redshifts.min():
                izLow = np.argwhere(self.redshifts<zmin).max()
        redshiftRange = np.arange(izLow,izUpp+1)
        # Determine luminosity function integral as function of redshift
        if verbose:
            print(funcname+"(): Integrating luminosity function for Galacticus snapshot outputs...")
        # Clear redshift integral ready for calculation
        self.redshiftIntegral = np.zeros_like(self.redshifts)
        dummy = [ self._integrateLuminosityFunctionAtRedshift(iz,datasetName,faintLimit,brightLimit=brightLimit,cumulative=cumulative,**kwargs)\
                      for iz in redshiftRange ]
        del dummy
        redshifts = np.copy(self.redshifts[redshiftRange])
        integral = np.copy(self.redshiftIntegral[redshiftRange])
        # Clear redshift integral array
        self.redshiftIntegral = np.zeros_like(self.redshifts)
        return redshifts,integral









class fluxNumberCounts(NumberCounts):
    
    def __init__(self,luminosityFunctionObj,hubble=None):
        super(fluxNumberCounts,self).__init__(luminosityFunctionObj,hubble=hubble)
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise attribute used to store counts
        self.bins = None
        self.counts = None
        return
    

    def _countsPerBin(self,datasetName,iBin,zmin,zmax,cumulative=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        kwargsInterpolate = self._get_interpolate_keywords(**kwargs)
        kwargsIntegrate = self._get_integrate_keywords(**kwargs)
        binWidth = self.bins[1] - self.bins[0]
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):
                lowLimit = self.bins[iBin]
                uppLimit = lowLimit + binWidth
        else:
                uppLimit = self.bins[iBin]
                lowLimit = uppLimit + binWidth
        if cumulative:
            uppLimit = None
        zIntegral,lfIntegral = self.integrateLuminosityFunctions(datasetName,lowLimit,brightLimit=uppLimit,zmin=zmin,zmax=zmax,\
                                                                     cumulative=cumulative,verbose=False,**kwargs)
        fz = interp1d(np.array(zIntegral),np.array(lfIntegral),**kwargsInterpolate)
        self.counts[datasetName][iBin] = romberg(fz,zmin,zmax,**kwargsIntegrate)
        return


    def _datasetCounts(self,datasetName,zmin,zmax,cumulative=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if "PROG" in kwargs.keys():
            PROG = kwargs["PROG"]
        dummy = [self._countsPerBin(datasetName,iBin,zmin,zmax,cumulative=cumulative,**kwargs) \
                     for iBin in np.arange(len(self.bins))]
        if PROG is not None:
            PROG.increment()
            PROG.print_status_line(task=" dataset = "+datasetName)
        del dummy
        allsky = 4.0*Pi*(180.0/Pi)**2    
        self.counts[datasetName] /= allsky        
        #if not cumulative:
        #    binWidth = self.bins[1] - self.bins[0]
        #    self.counts[datasetName] /= binWidth
        return


    def computeCounts(self,datasets,bins,zmin=None,zmax=None,cumulative=False,verbose=False,**kwargs):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check datasets available in luminosity function file
        goodDatasets = list(set(datasets).intersection(self.availableDatasets))
        badDatasets = list(set(datasets).difference(self.availableDatasets))
        if len(badDatasets) > 0:
            msg = "WARNING! "+funcname+"(): Following datasets cannot be found and will be ignored..."
            msg = "\n        " + "\n        ".join(badDatasets)
            print(msg)
        # Construct array to store counts
        dtype = [(p,float) for p in goodDatasets]
        self.counts = np.zeros(len(bins),dtype=dtype)
        self.bins = bins
        # Set redshift limits
        if zmin is None:
            zmin = self.redshifts.min()
        else:
            zmin = np.maximum(self.redshifts.min(),zmin)        
        if zmax is None:
            zmax = self.redshifts.max()
        else:
            zmax = np.minimum(self.redshifts.max(),zmax)                            
        # Compute counts
        if verbose:
            print(funcname+"(): Computing number counts between z = "+str(zmin)+" and z = "+str(zmax)+" ...")
            PROG = Progress(len(goodDatasets))
        else:
            PROG = None
        dummy = [self._datasetCounts(name,zmin,zmax,PROG=PROG,cumulative=cumulative,**kwargs) \
                     for name in goodDatasets]
        del dummy
        counts = np.copy(self.counts)
        self.bins = None
        self.counts = None
        return bins,counts


    def writeToHDF5(self,ofile,bins,counts,cumulative=False,zmin=None,zmax=None,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        fileObj = HDF5(ofile,'w')
        # Write value of cosmological parameters
        fileObj.mkGroup("Cosmology")
        dummy = [fileObj.addAttributes("Cosmology",{key:self.luminosityFunction.cosmology[key]}) \
                     for key in self.luminosityFunction.cosmology.keys()]
        # Specify cumulative attribute
        fileObj.addAttributes("/",{"cumulative":str(int(cumulative))})
        # Write bins
        fileObj.addDataset("/","countsBins",bins,chunks=True,compression="gzip",\
                               compression_opts=6)        
        if len(fnmatch.filter(counts.dtype.names,"*LineLuminosity*"))>0:
            fileObj.addAttributes("/countsBins",{"units":"log10(erg/s/cm^2)"})
        # Create counts group
        fileObj.mkGroup("counts")        
        # Specify cumulative attribute
        fileObj.addAttributes("/counts",{"cumulative":str(int(cumulative))})
        if cumulative:            
            fileObj.addAttributes("/counts",{"units":"/deg^2"})
        else:
            if len(fnmatch.filter(counts.dtype.names,"*LineLuminosity*"))>0:
                fileObj.addAttributes("/counts",{"units":"/logflux/deg2"})
            else:
                fileObj.addAttributes("/counts",{"units":"/mag/deg2"})          
        # Specify redshift range
        if zmin is None:
            zmin = "unknown"
        fileObj.addAttributes("/counts",{"zmin":str(zmin)})
        if zmax is None:
            zmax = "unknown"
        fileObj.addAttributes("/counts",{"zmax":str(zmax)})
        # Write properties
        dummy = [fileObj.addDataset("/counts",name,np.copy(counts[name]),chunks=True,compression="gzip",\
                               compression_opts=6) for name in counts.dtype.names]
        del dummy
        fileObj.close()
        if verbose:
            print(funcname+"(): counts written to file "+ofile)
        return



class redshiftNumberCounts(NumberCounts):
    
    def __init__(self,luminosityFunctionObj):
        super(redshiftNumberCounts,self).__init__(luminosityFunctionObj)
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise attribute used to store counts
        self.bins = None
        self.counts = None
        return


    def _computeCountsForDataset(self,datasetName,faintLimit,brightLimit=None,verbose=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        zIntegral,phiIntegral = self.integrateLuminosityFunctions(datasetName,\
                                                                      faintLimit,brightLimit=brightLimit,\
                                                                      zmax=self.bins.max(),cumulative=False,\
                                                                      verbose=False,**kwargs)                        
        kwargsInterpolate = self._get_interpolate_keywords(**kwargs)    
        fz = interp1d(np.array(zIntegral),np.array(phiIntegral),**kwargsInterpolate)
        self.counts[datasetName] = fz(self.bins)
        allsky = 4.0*Pi*(180.0/Pi)**2
        self.counts[datasetName] /= allsky
        if "PROG" in kwargs.keys():
            PROG = kwargs["PROG"]
            if PROG is not None:
                PROG.increment()
                PROG.print_status_line(task=" dataset = "+datasetName)
        return 
    
    def computeCounts(self,datasets,bins,faintLimit,brightLimit=None,verbose=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check datasets available in luminosity function file
        goodDatasets = list(set(datasets).intersection(self.availableDatasets))
        badDatasets = list(set(datasets).difference(self.availableDatasets))
        if len(badDatasets) > 0:
            msg = "WARNING! "+funcname+"(): Following datasets cannot be found and will be ignored..."
            msg = "\n        " + "\n        ".join(badDatasets)
            print(msg)
        # Construct array to store counts
        dtype = [(p,float) for p in goodDatasets]
        self.counts = np.zeros(len(bins),dtype=dtype)
        self.bins = bins
        # Loop over datasets
        PROG = Progress(len(goodDatasets))
        dummy = [ self._computeCountsForDataset(name,faintLimit,brightLimit=brightLimit,verbose=verbose,PROG=PROG,**kwargs)\
                      for name in goodDatasets]
        del dummy
        # Extract values and exit
        bins = np.copy(self.bins)
        counts = np.copy(self.counts)
        self.bins = None
        self.counts = None        
        return bins,counts




