#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from .io import GalacticusHDF5
from .EmissionLines import GalacticusEmissionLine,GalacticusEmissionLines
from .Luminosities import ergPerSecond
from .constants import erg,luminosityAB,luminositySolar,jansky,kilo
from .constants import angstrom,megaParsec,Pi,speedOfLight,centi
from .constants import plancksConstant
from .Inclination import getInclination
from .GalacticusErrors import ParseError
from .statistics.utils import mad


def sedStatistic(seds,axis=0,statistic="mean"):
    if statistic.lower() == "mean": 
        stat = np.mean(np.log10(seds),axis=axis)
        stat = 10.0**stat
    elif fnmatch.fnmatch(statistic.lower(),"std*"):
        stat = np.std(np.log10(seds),axis=axis)
        stat = 10.0**stat
    elif fnmatch.fnmatch(statistic.lower(),"med*"):
        stat = np.median(seds,axis=axis)
    elif statistic.lower() == "mad": 
        stat = np.mad(seds,axis=axis)
    else:
        stat = None
    return stat


def normaliseSED(wavelength,sed,normWavelength,**kwargs):    
    funcname = sys._getframe().f_code.co_name
    if normWavelength < wavelength.min() or normWavelength > wavelength.max():
        raise ValueError(funcname+"(): normalisation point outside wavelength range, ("+\
                             str(wavelength.min())+","+str(wavelength.max())+")")
    kwaargs["axis"] = 0
    f = interp1d(wavelength,sed,**kwargs)
    norm = f(normWavelength)
    return sed/norm

def interpolateSED(wavelengths,seds,newWavelength,**kwargs):
    funcname = sys._getframe().f_code.co_name
    kwargs["axis"] = 1
    f = interp1d(wavelengths,seds,**kwargs)
    return f(newWavelength)



class sedTopHatArray(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        """
        sedTopHatArray: Class to store information for array of SED top hat filters included in Galacticus HDF5 output.

        USAGE:  THA = sedTophatArray(galHDF5Obj,z,[verbose])
        
            INPUTS:
            galHDF5Obj : GalacticusHDF5 object
            z             : redshift
            verbose       : Print additional information (T/F, default=F)

            OUTPUTS:
            THA           : sedTopHatArray object.

        """
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        # Initialise variables to store top hat information
        self.redshift = None
        self.topHats = None
        self.filters = None        
        self._filterOrder = None
        self.wavelengths = None
        return


    def reset(self):
        self.redshift = None
        self.topHats = None
        self.filters = None        
        self._filterOrder = None
        self.wavelengths = None
        return

    def setHDF5Output(self,z):
        self.redshift = z
        # Store list of all possible top hat filters in specified output of HDF5 file
        self.topHats = fnmatch.filter(self.galHDF5Obj.availableDatasets(self.redshift),"*LuminositiesStellar:sedTopHat*")	
        # Get list of filter wavelengths and widths
        regexString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:"+\
                       "sedTopHat_([\d\.]+)_([\d\.]+)"+\
                       ":(?P<frame>[^:]+)"+\
                       "(:z(?P<redshift>[\d\.]+))"+\
                       "(?P<recent>:recent)?"+\
                       "(?P<dust>:dust[^:]+)?$"
        MATCH = re.search(regexString,self.topHats[0])
        recent = MATCH.group('recent')
        if not recent:
            recent = ""
        dust = MATCH.group('dust')
        if not dust:
            dust = ""            
        searchString = MATCH.group('component')+"LuminositiesStellar:sedTopHat_*_*:"+MATCH.group('frame')+":z"+MATCH.group('redshift')+recent+dust
        prefix = searchString.split(":")[0] + ":sedTopHat_"
        suffix = ":"+":".join(searchString.split(":")[2:])
        topHats = " ".join(fnmatch.filter(self.topHats,searchString)).replace(prefix,"").replace(suffix,"")
        self.filters = [(f.split("_")[0],f.split("_")[1]) for f in topHats.split()]
        # Get order of filters in increaasing wavelength
        self._filterOrder = np.argsort([float(f[0]) for f in self.filters])
        # Store wavelengths
        self.wavelengths = np.array([float(f[0]) for f in self.filters])[self._filterOrder]
        return

    def getLuminosity(self,filterName,selectionMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        """
        sedTopHatArray.getLuminosity: Return luminosities for specified top hat filter. Includes ability to provide a
                                      mask to select only subset of galaxies.

        USAGE: luminosity = sedTopHatArray.getLuminosity(filterName,[selectionMask])
        
           INPUTS:
           filterName    : Name of top hat filter to extract.
           selectionMask : Logical mask to select subset of galaxies.
        
           OUTPUTS:
           luminosity    : Luminosity of galaxies in specified top hat filter.
        
        """
        # Get output for redshift
        out = self.galHDF5Obj.selectOutput(float(self.redshift))
        # Get stellar luminosity
        luminosity = np.array(out["nodeData/"+filterName])
        if selectionMask is not None:
            luminosity = luminosity[selectionMask]
        return luminosity

    def getLuminosities(self,component,frame,dust="",recent="",selectionMask=None):
        """
        sedTopHatArray.getLuminosities: Return luminosities for top hat filter set.

        USAGE:  luminosities = sedTopHatArray.getLuminosities(component,frame,[dust],[recent],[selectionMask])

             INPUTS:
             component       : Component of galaxy ('disk', 'spheroid', or 'total')
             frame           : Frame of reference ('rest' or 'observed')
             dust            : Dust information for filter name (e.g. ':dustAtlas')
             recent          : Consider only recent star formation. Specify as recent=':recent'.
             selectionMask   : Logical mask to select subset of galaxies.

             OUTPUTS:
             luminosities    : 2D array of luminosities of size (nfilters x ngalaxies).
        
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = component+"LuminositiesStellar:sedTopHat_*_*:"+frame+":"+self.galHDF5Obj.getRedshiftString(self.redshift)+dust+recent
        if fnmatch.fnmatch(component,"total"):
            luminosities = self.getLuminosities("disk",frame,dust=dust,recent=recent,selectionMask=selectionMask)+\
                           self.getLuminosities("spheroid",frame,dust=dust,recent=recent,selectionMask=selectionMask)
        else:                        
            filters = np.array(fnmatch.filter(self.topHats,searchString))[self._filterOrder]
            luminosities = [self.getLuminosity(filterName,selectionMask=selectionMask)\
                           for filterName in filters]
            luminosities = np.stack(np.copy(luminosities),axis=1)
        return luminosities
        


class GalacticusSED(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.verbose = verbose  
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj
        # Initialise top hat array object
        self.topHatArray = sedTopHatArray(self.galHDF5Obj)
        # Initialise emission lines object
        self.EmissionLines = EmissionLineProfiles(self.galHDF5Obj)
        # Initialise variables to store HDF5 output information
        self.hdf5Output = None
        self.hdf5Redshift = None
        self.haveLightconeRedshifts = None
        self.totalNumberGalaxies = 0
        # Initialise variables to store SED and galaxy information
        self.datasetName = None
        self.wavelengths = None
        self.galaxyRedshift = None
        self.galaxySED = None
        self.galaxyMask = None
        return
            
    def resetHDF5Output(self):
        self.topHatArray.reset()
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        self.haveLightconeRedshifts = None
        self.totalNumberGalaxies = 0
        return

    def setHDF5Output(self,z):
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        self.topHatArray.setHDF5Output(self.redshift)
        self.totalNumberGalaxies = self.galHDF5Obj.countGalaxiesAtRedshift(float(self.redshift))
        if self.totalNumberGalaxies > 0:
            self.haveLightconeRedshifts = self.galHDF5Obj.datasetExists("lightconeRedshift",self.redshift)        
            if self.haveLightconeRedshifts:
                self.galaxyRedshift = np.array(self.hdf5Output["nodeData/lightconeRedshift"])
            else:
                self.galaxyRedshift = np.ones(self.totalNumberGalaxies,dtype=float)*float(self.redshift)
        return

    def resetSED(self):
        self.wavelengths = None
        self.galaxyRedshift = None
        self.galaxySED = None
        self.galaxyMask = None
        return

    def resetDatasetName(self):
        self.datasetName = None
        return
        
    def initialiseSED(self,wavelengths,z=None,galaxyMask=None):
        if z is not None:
            self.setHDF5Output(z)
        self.wavelengths = wavelengths
        if galaxyMask is None:
            self.galaxyMask = np.ones(self.totalNumberGalaxies,dtype=bool)
        else:
            self.galaxyMask = galaxyMask
        self.galaxySED = np.zeros((np.count_nonzero(self.galaxyMask),len(self.wavelengths)),dtype=float)
        return
        
    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^(?P<component>disk|spheroid|total)SED:"+\
                       "(?P<frame>[^:]+)"+\
                       "(?P<emlines>:(?P<lineProfile>[^:_]+)_(?P<lineWidth>[^:]+))?"+\
                       "(?P<snrString>:snr(?P<snr>[\d\.]+))?"+\
                       "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
                       "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
        self.datasetName = re.search(searchString,datasetName)
        # Check dataset name corresponds to an SED
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return

    def addContinuumNoise(self,wavelengths,luminosities,SNR):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get Poisson error on count rate of photons
        energy = speedOfLight*plancksConstant/np.stack([wavelengths]*luminosities.shape[0])*angstrom
        counts = luminosities*luminosityAB/energy
        # Perturb counts and convert back to luminosities
        counts = norm.rvs(loc=counts,scale=counts/SNR)
        return counts*energy/luminosityAB
        
    def getContinuumSED(self,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract necessary information
        component = self.datasetName.group('component')
        frame = self.datasetName.group('frame')        
        snr = self.datasetName.group('snr')
        redshift = self.datasetName.group('redshift')
        dust = self.datasetName.group('dust')
        if dust is None:
            dust = ""
        recent = self.datasetName.group('recent')
        if recent is None:
            recent = ""
        # Compute continuum SED
        sed = interpolateSED(self.topHatArray.wavelengths,
                             self.topHatArray.getLuminosities(component,frame,dust=dust,\
                                                              recent=recent,selectionMask=self.galaxyMask),\
                             self.wavelengths,axis=1,**kwargs)
        # Add noise?
        if snr is not None:
            sed = self.addContinuumNoise(self.wavelengths,sed,float(snr))    
        return sed

    def updateResolution(self,wavelengthInterval):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        minWavelength = self.wavelengths.min()
        maxWavelength = self.wavelengths.max()    
        newWavelengths = np.linspace(minWavelength,maxWavelength,len(np.arange(minWavelength,maxWavelength,wavelengthInterval))+1)    
        self.sed = interpolateSED(self.wavelengths,self.sed,newWavelengths,axis=1)
        self.wavelengths = newWavelengths
        return
    
    def ergPerSecond(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxySED = np.log10(self.galaxySED)
        self.galaxySED += np.log10(luminosityAB)
        self.galaxySED -= np.log10(erg)
        self.galaxySED = 10.0**self.galaxySED
        return

    def buildSED(self,datasetName,wavelengths,z=None,galaxyMask=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set dataset name
        self.setDatasetName(datasetName)
        # Initialise SED        
        self.initialiseSED(wavelengths,z=z,galaxyMask=galaxyMask)
        # Get continuum luminosity
        self.galaxySED = self.getContinuumSED(**kwargs)
        # Introduce emission lines
        if self.datasetName.group('emlines'):
            self.EmissionLines.reset()
            self.EmissionLines.buildLineProfiles(datasetName,self.wavelengths,z=self.redshift,galaxyMask=self.galaxyMask)
            self.galaxySED += self.EmissionLines.profileSum
        # Convert units to microJanskys
        self.ergPerSecond()
        comDistance = self.galHDF5Obj.cosmology.comoving_distance(self.redshift)*megaParsec/centi
        self.galaxySED /= 4.0*Pi*comDistance**2
        self.galaxySED /= jansky
        self.galaxySED *= 1.0e6
        ## Compute a statistic if specified
        #if statistic is not None:
        #    sed = sedStatistic(sed,axis=0,statistic=statistic)
        return
        
    def getSED(self):
        return self.wavelengths,self.galaxySED


class EmissionLineProfiles(object):
    
    def __init__(self,galHDF5Obj,verbose=False,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj        
        # Create emission lines object
        self.EmissionLines = GalacticusEmissionLine(self.galHDF5Obj,**kwargs)                
        # Initialise HDF5 object variables
        self.redshift = None
        self.hdf5Output = None
        # Initialise variables to store SED and addtional galaxy specifications        
        self.datasetName = None
        self.sedWavelengths = None
        self.profileSum = None
        self.galaxyMask = None
        self.galaxyRedshift = None        
        return

    def reset(self):
        self.datasetName = None
        self.sedWavelengths = None
        self.profileSum = None
        self.galaxyMask = None
        self.galaxyRedshift = None        
        return

    def resetHDF5Output(self):
         self.redshift = None
         self.hdf5Output = None
         return

    def setHDF5Output(self,z):
        # Store redshift
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        # Store redshift output
        self.hdf5Output = self.galHDF5Obj.selectOutput(float(self.redshift))
        # Set redshifts of galaxies
        if self.galHDF5Obj.datasetExists("lightconeRedshift",self.redshift):
            self.galaxyRedshift = np.array(self.hdf5Output["nodeData/lightconeRedshift"])
        else:
            self.galaxyRedshift = np.ones(self.galHDF5Obj.countGalaxiesAtRedshift(self.redshift))*self.redshift
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        searchString = "^(?P<component>disk|spheroid|total)SED:"+\
                       "(?P<frame>[^:]+)"+\
                       "(?P<emlines>:(?P<lineProfile>[^:_]+)_(?P<lineWidth>[^:]+))?"+\
                       "(?P<snrString>:snr(?P<snr>[\d\.]+))?"+\
                       "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
                       "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
        self.datasetName = re.search(searchString,datasetName)
        # Check dataset name corresponds to an SED
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return

    def setSedSpecifications(self,wavelengths,galaxyMask=None):
        # Store wavelengths
        self.sedWavelengths = wavelengths
        # Store selection mask to apply to galaxies
        if galaxyMask is None:
            self.galaxyMask = self.np.ones(self.galHDF5.countGalaxiesAtRedshift(self.redshift),astype=bool)
        else:                
            self.galaxyMask = galaxyMask             
        self.profileSum = np.zeros((np.sum(self.galaxyMask),len(self.sedWavelengths)),dtype=np.float64)
        return

    def buildLineProfiles(self,datasetName,wavelengths,z=None,galaxyMask=None):
        self.reset()
        self.setDatasetName(datasetName)
        self.setHDF5Output(z)
        self.setSedSpecifications(wavelengths,galaxyMask=galaxyMask)        
        self.sumLineProfiles()
        return 

    
    def sumLineProfiles(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sum profiles
        dummy = [self.addLine(name) for name in self.EmissionLines.getLineNames()]
        # Return sum
        frequency = speedOfLight/np.stack([self.sedWavelengths]*self.profileSum.shape[0])*angstrom
        self.profileSum /= frequency
        return 


    def addLine(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset information
        component = self.datasetName.group('component')
        frame = self.datasetName.group('frame')
        lineProfile = self.datasetName.group('lineProfile')
        lineWidth = self.datasetName.group('lineWidth')
        redshift = self.datasetName.group('redshift')
        dust = self.datasetName.group('dust')
        if not dust:
            dust = ""
        recent = self.datasetName.group('recent')
        if not recent:
            recent = ""
        # Get observed wavelength of line 
        lineWavelength = self.EmissionLines.getWavelength(lineName)
        if fnmatch.fnmatch(frame,"observed"):
            lineWavelength *= (1.0+self.galaxyRedshift)
        else:
            lineWavelength = np.ones_like(self.galaxyRedshift)*lineWavelength
        lineWavelength = lineWavelength[self.galaxyMask]
        # Get line luminosity
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+recent+dust
        if lineDatasetName not in self.hdf5Output["nodeData"].keys():
            print("WARNING! "+funcname+"(): Cannot locate '"+lineDatasetName+"' for inclusion in SED. Will be skipped.")
            return 
        lineLuminosity = np.array(self.hdf5Output["nodeData/"+lineDatasetName])[self.galaxyMask]           
        np.place(lineLuminosity,lineLuminosity<0.0,0.0)
        # Compute FWHM (use fixed line width in km/s)
        FWHM = self.getFWHM(lineName,lineWidth)
        # Compute line profile
        if fnmatch.fnmatch(lineProfile.lower(),"gaus*"):
            self.profileSum += self.gaussian(lineWavelength,lineLuminosity,FWHM)
        elif fnmatch.fnmatch(lineProfile.lower(),"voigt*"):
            self.profileSum += self.cauchy(lineWavelength,lineLuminosity,FWHM)
        else:
            raise ValueError(funcname+"(): line profile must be Gaussian or Voigt! Other profiles not yet implemented!")
        return 


    def getApproximateVelocityDispersion(self,scaleVelocityRatio=0.1):
        """
        
        diskVelocityDispersion = diskVelocity * SQRT(sin(inclination)**2+(scaleVelocityRatio*cos(inclination))**2)

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Approximate spheroid velocity dispersion using spheroid 'rotation velocity'
        approximateVelocityDispersion = np.copy(np.array(self.hdf5Output["nodeData/spheroidVelocity"]))
        # Determine spheroid-to-total mass ratio
        baryonicSpheroidMass = np.copy(np.array(self.hdf5Output["nodeData/spheroidMassStellar"])) + \
            np.copy(np.array(self.hdf5Output["nodeData/spheroidMassGas"]))
        baryonicDiskMass = np.copy(np.array(self.hdf5Output["nodeData/diskMassStellar"])) + \
            np.copy(np.array(self.hdf5Output["nodeData/diskMassGas"]))
        baryonicSpheroidToTotalRatio = baryonicSpheroidMass/(baryonicSpheroidMass+baryonicDiskMass)
        # Check if any disk-dominated galaxies in dataset and replace corresponding velocities
        diskDominated = baryonicSpheroidToTotalRatio<0.5
        if any(diskDominated):
            # Approximate disk velocity dispersion using combiantion of disk rotational velocity
            # and disk vertical velocity (computed as fraction of rotation velocity)
            diskVelocity = np.copy(np.array(self.hdf5Output["nodeData/diskVelocity"]))
            inclination = getInclination(self.galHDF5Obj,float(self.redshift))*Pi/180.0
            diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)            
            np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
        return approximateVelocityDispersion[self.galaxyMask]


    def getFWHM(self,lineName,lineWidth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract rest wavelength
        restWavelength = self.EmissionLines.getWavelength(lineName)
        # Compute FWHM        
        if fnmatch.fnmatch(lineWidth.lower(),"fixedwidth*"):                        
            widthVelocity = np.ones_like(self.profileSum)*float(lineWidth.replace("fixedWidth",""))
        elif lineWidth.lower() == "velocityapprox":
            widthVelocity = self.getApproximateVelocityDispersion(scaleVelocityRatio=0.1)
            widthVelocity = np.stack([widthVelocity]*len(self.sedWavelengths),axis=1).reshape(len(widthVelocity),-1)
        else:
            raise ValueError(funcname+"(): line width method must be 'fixed'! Other methods not yet implemented!")
        c = speedOfLight/kilo
        FWHM = restWavelength*(widthVelocity/c)            
        return FWHM
    

    def gaussian(self,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        sigma = FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))
        # Compute amplitude for Gaussian
        amplitude = np.stack([lineLuminosity]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= (sigma*np.sqrt(2.0*Pi))        
        # Compute luminosity
        wavelengths = np.concatenate([self.sedWavelengths]*len(lineLuminosity)).reshape(-1,len(self.sedWavelengths))
        lineWavelengths = np.stack([lineWavelength]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        luminosity = amplitude*np.exp(-((wavelengths-lineWavelengths)**2)/(2.0*(sigma**2)))
        return luminosity


    def voigt(self,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        gamma = FWHM/2.0
        # Compute amplitude for Voigt profile
        amplitude = np.stack([lineLuminosity]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= Pi*gamma
        # Compute luminosity
        wavelengths = np.concatenate([self.sedWavelengths]*len(lineLuminosity)).reshape(-1,len(self.sedWavelengths))
        lineWavelengths = np.stack([lineWavelength]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                    
        luminosity = amplitude*(gamma**2)/(((wavelengths-lineWavelengths)**2)+gamma**2)
        return luminosity



