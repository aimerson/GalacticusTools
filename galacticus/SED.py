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
    
    def __init__(self,galObj,z,verbose=False):
        """
        sedTopHatArray: Class to store information for array of SED top hat filters included in Galacticus HDF5 output.

        USAGE:  THA = sedTophatArray(galacticusOBJ,z,[verbose])
        
            INPUTS:
            galacticusOBJ : GalacticusHDF5 object
            z             : redshift
            verbose       : Print additional information (T/F, default=F)

            OUTPUTS:
            THA           : sedTopHatArray object.

        """
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galacticusOBJ = galObj
        self.redshift = z
        # Store list of all possible top hat filters in specified output of HDF5 file
        self.topHats = fnmatch.filter(self.galacticusOBJ.availableDatasets(self.redshift),"*LuminositiesStellar:sedTopHat*")	
        # Get list of filter wavelengths and widths
        MATCH = re.search(r"^(disk|spheroid|total)LuminositiesStellar:sedTopHat_([\d\.]+)_([\d\.]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:recent)?",self.topHats[0])
        searchString = MATCH.group(1)+"LuminositiesStellar:sedTopHat_*_*:"+MATCH.group(4)+":z"+MATCH.group(5)
        if MATCH.group(6) is not None:
            searchString = searchString+MATCH.group(6)
        if MATCH.group(7) is not None:
            searchString = searchString+MATCH.group(7)
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
        out = self.galacticusOBJ.selectOutput(float(self.redshift))
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
        searchString = component+"LuminositiesStellar:sedTopHat_*_*:"+frame+":"+self.galacticusOBJ.getRedshiftString(self.redshift)+dust+recent
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
    
    def __init__(self,galObj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.verbose = verbose  
        # Store GalacticusHDF5 object
        self.galacticusOBJ = galObj
        # Initialise emission lines object
        self.EmissionLines = GalacticusEmissionLines()
        # Initialise variables to store SED information
        self.datasetName = None
        self.wavelengths = None
        self.sed = None
        self.redshift = None
        self.MASK = None
        return
        
    def reset(self):
        self.datasetName = None
        self.wavelengths = None
        self.sed = None
        self.redshift = None
        self.MASK = None
        return

    def setSelectionMask(self,mask):
        self.MASK = mask
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.datasetName = re.search(r"^(disk|spheroid|total)SED:([^:]+)?(:[^:]+)??(:snr[\d\.]+)?:z([\d\.]+)(:dust[^:]+)?(:recent)?",datasetName)
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
        

    def getContinuumSED(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract necessary information
        component = self.datasetName.group(1)
        frame = self.datasetName.group(2)        
        snr = self.datasetName.group(4)
        if snr is not None:
            snr = snr.replace(":snr","")
        redshift = self.datasetName.group(5)
        dust = self.datasetName.group(6)
        if dust is None:
            dust = ""
        option = self.datasetName.group(7)
        if option is None:
            option = ""
        # Get top hat luminosities
        sTHA = sedTopHatArray(self.galacticusOBJ,float(redshift),verbose=self.verbose)
        # Store additional SED information
        self.redshift = float(redshift)
        self.wavelengths = sTHA.wavelengths
        # Compute continuum SED
        self.sed = sTHA.getLuminosities(component,frame,dust=dust,recent=option,selectionMask=self.MASK)
        # Add noise?
        if snr is not None:
            self.sed = self.addContinuumNoise(self.wavelengths,self.sed,float(snr))    
        return

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
        self.sed = np.log10(self.sed)
        self.sed += np.log10(luminosityAB)
        self.sed -= np.log10(erg)
        self.sed = 10.0**self.sed
        return

    def buildSED(self,datasetName,wavelengthInterval,selectionMask=None,statistic=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set dataset name
        self.setDatasetName(datasetName)
        # Set selection mask
        self.setSelectionMask(selectionMask)
        # Get continuum luminosity
        self.getContinuumSED()
        # Update resolution
        if False:
            self.updateResolution(newResolution)
        ## Introduce emission lines
        if self.datasetName.group(3):
            frame = self.datasetName.group(2)
            redshift = float(self.datasetName.group(5))
            LINES = EmissionLineProfiles(self.galacticusOBJ,frame,redshift,self.wavelengths,\
                                         selectionMask=selectionMask,verbose=self.verbose)
            if not len(LINES.linesInRange) == 0:
                self.sed += LINES.sumLineProfiles(self.datasetName,ignoreResolution=ignoreResolution)
                self.sed = np.maximum(self.sed,LINES.sumLineProfiles(MATCH,profile='gaussian'))
        # Convert units to microJanskys
        self.ergPerSecond()
        comDistance = self.galacticusOBJ.cosmology.comoving_distance(self.redshift)*megaParsec/centi
        self.sed /= 4.0*Pi*comDistance**2
        self.sed /= jansky
        self.sed *= 1.0e6
        # Compute a statistic if specified
        if statistic is not None:
            sed = sedStatistic(sed,axis=0,statistic=statistic)
        return
        
    def getSED(self):
        return self.wavelengths,self.sed



class spectralEnergyDistribution(object):
    
    def __init__(self,galObj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galacticusOBJ = galObj
        self.EmissionLines = GalacticusEmissionLines()
        self.verbose = verbose        
        return

        
    def interpolateContinuum(self,wavelengths,luminosities,wavelengthDifference=10.0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        f = interp1d(wavelengths,luminosities,axis=1)
        newWavelengths = np.arange(wavelengths.min(),wavelengths.max(),wavelengthDifference)
        newLuminosities = f(newWavelengths)
        return newWavelengths,newLuminosities

    
    def addContinuumNoise(self,wavelengths,luminosities,SNR):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get Poisson error on count rate of photons
        energy = speedOfLight*plancksConstant/np.stack([wavelengths]*luminosities.shape[0])*angstrom
        counts = luminosities*luminosityAB/energy
        # Perturb counts and convert back to luminosities
        counts = norm.rvs(loc=counts,scale=counts/SNR)
        return counts*energy/luminosityAB


    def ergPerSecond(self,luminosity):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        luminosity = np.log10(luminosity)
        luminosity += np.log10(luminosityAB)
        luminosity -= np.log10(erg)
        luminosity = 10.0**luminosity
        return luminosity

    def getSED(self,datasetName,selectionMask=None,ignoreResolution=False,statistic=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset name correspnds to an SED
        MATCH = re.search(r"^(disk|spheroid|total)SED:([^:]+)?(:[^:]+)??(:snr[\d\.]+)?:z([\d\.]+)(:dust[^:]+)?(:recent)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract necessary information
        component = MATCH.group(1)
        frame = MATCH.group(2)        
        lineInformation = MATCH.group(3)        
        if lineInformation is None:
            lineInformation = ""
            includeEmissionLines = False
        else:
            includeEmissionLines = True
        snr = MATCH.group(4)
        if snr is not None:
            snr = snr.replace(":snr","")
        redshift = MATCH.group(5)
        dust = MATCH.group(6)
        if dust is None:
            dust = ""
        option = MATCH.group(7)
        if option is None:
            option = ""
        # Create class of top hat filters
        sTHA = sedTopHatArray(self.galacticusOBJ,float(redshift),verbose=self.verbose)
        sed = sTHA.getLuminosities(component,frame,dust=dust,recent=option,selectionMask=selectionMask)
        # Add noise?
        if snr is not None:
            sed = self.addContinuumNoise(sTHA.wavelengths,sed,float(snr))
        # Introduce emission lines
        if includeEmissionLines:
            LINES = EmissionLineProfiles(self.galacticusOBJ,frame,redshift,sTHA.wavelengths,\
                                             selectionMask=selectionMask,verbose=self.verbose)
            if not len(LINES.linesInRange) == 0:
                sed += LINES.sumLineProfiles(MATCH,ignoreResolution=ignoreResolution)
                #sed = np.maximum(sed,LINES.sumLineProfiles(MATCH,profile='gaussian'))
        # Convert units to microJanskys
        sed = self.ergPerSecond(sed)
        comDistance = self.galacticusOBJ.cosmology.comoving_distance(float(redshift))*megaParsec/centi
        sed /= 4.0*Pi*comDistance**2
        sed /= jansky
        sed *= 1.0e6
        # Compute a statistic if specified
        if statistic is not None:
            sed = sedStatistic(sed,axis=0,statistic=statistic)
        return wavelengths,sed
        
        
        

class EmissionLineProfiles(object):
    
    def __init__(self,galHDF5Obj,verbose=False,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        # Store GalacticusHDF5 object
        self.galHDF5OBJ = galHDF5Obj        
        # Create emission lines object
        self.EmissionLines = GalacticusEmissionLine(**kwargs)                
        # Initialise HDF5 object variables
        self.redshift = None
        self.hdf5Output = None
        # Initialise variables to store SED and addtional galaxy specifications        
        self.frame = None
        self.sedWavelegnths = None
        self.selectionMask = None
        self.galaxyRedshift = None        
        self.linesObservable = None
        return

    def setHDF5Output(self,z):
        # Store redshift
        self.redshift = z
        # Store redshift output
        self.hdf5Output = self.galHDF5Obj.selectOutput(float(self.redshift))        
        return

    def setSedSpecifications(self,wavelengths,frame,galaxyRedshift,selectionMask=None):
        # Store frame
        self.frame = frame
        # Store wavelengths
        self.sedWavelengths = wavelengths
        # Store redshifts of galaxies
        self.galaxyRedshift = galaxyRedshift
        # Store selection mask to apply to galaxies
        self.selectionMask = selectionMask        
        return

    



        # Check for any emission lines inside wavelength range
        lines = np.array(self.EmissionLines.getLineNames())
        if self.frame == "rest":
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name) for name in lines])
        else:
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name,redshift=float(redshift)) for name in lines])
        emissionLinesInRange = np.logical_and(effectiveWavelengths>=self.sedWavelengths.min(),effectiveWavelengths<=self.sedWavelengths.max())
        self.linesInRange = lines[emissionLinesInRange]
        # Exit here if no lines in range
        if len(self.linesInRange) == 0:
            return
        # Store wavelengths the lines appear at
        self.wavelengthsInRange = effectiveWavelengths[emissionLinesInRange]


        # Create 2D array to store sum of line profiles 
        if self.selectionMask is None:
            ngals = len(np.array(self.OUT["nodeData/nodeIndex"]))
        else:
            ngals = len(np.array(self.OUT["nodeData/nodeIndex"])[self.selectionMask])
        self.profileSum = np.zeros((ngals,len(self.sedWavelengths)),dtype=np.float64)
        return

    
    def sumLineProfiles(self,MATCH,ignoreResolution=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sum profiles
        dummy = [self.addLine(lineName,MATCH,ignoreResolution=ignoreResolution) for lineName in self.linesInRange]
        # Return sum
        frequency = speedOfLight/np.stack([self.sedWavelengths]*self.profileSum.shape[0])*angstrom
        self.profileSum /= frequency
        return self.profileSum


    def addLine(self,lineName,MATCH,ignoreResolution=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset information
        component = MATCH.group(1)
        resolution = MATCH.group(2)
        frame = MATCH.group(3)
        lineInformation = MATCH.group(4)
        snr = MATCH.group(5)
        if snr is not None:
            snr = snr.replace(":snr","")
        redshift = MATCH.group(6)
        dust = MATCH.group(7)
        if dust is None:
            dust = ""
        option = MATCH.group(8)
        if option is None:
            option = ""
        # Extra line profile information
        INFO = re.search(r"^:([^_]+)_([^_]+)",lineInformation)
        profile = INFO.group(1)
        lineWidth = INFO.group(2)
        # Get dataset name
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+dust+option
        if lineDatasetName not in self.OUT["nodeData"].keys():
            print("WARNING! "+funcname+"(): Cannot locate '"+lineDatasetName+"' for inclusion in SED. Will be skipped.")
            return
        # Get observed wavelength of line 
        iline = np.argwhere(self.linesInRange==lineName)[0]
        lineWavelength = self.wavelengthsInRange[iline][0]
        # Extract line luminosity
        lineLuminosity = np.array(self.OUT["nodeData/"+lineDatasetName])
        if self.selectionMask is not None:
            lineLuminosity = lineLuminosity[self.selectionMask]
        np.place(lineLuminosity,lineLuminosity<0.0,0.0)
        # Compute FWHM (use fixed line width in km/s)
        FWHM = self.getFWHM(lineName,lineWidth)
        # Check if FWHM larger than resolution (if resolution specified)
        if not ignoreResolution:
            resolutionLimit = lineWavelength/float(resolution)
            FWHM = np.maximum(FWHM,resolutionLimit)
        # Compute line profile
        if fnmatch.fnmatch(profile.lower(),"gaus*"):
            self.profileSum += self.gaussian(lineWavelength,lineLuminosity,FWHM)
        elif fnmatch.fnmatch(profile.lower(),"cauchy*"):
            self.profileSum += self.cauchy(lineWavelength,lineLuminosity,FWHM)
        else:
            raise ValueError(funcname+"(): line profile must be Gaussian or Cauchy! Other profiles not yet implemented!")
        return 


    def getApproximateVelocityDispersion(self,scaleVelocityRatio=0.1):
        """
        
        diskVelocityDispersion = diskVelocity * SQRT(sin(inclination)**2+(scaleVelocityRatio*cos(inclination))**2)

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Approximate spheroid velocity dispersion using spheroid 'rotation velocity'
        approximateVelocityDispersion = np.copy(np.array(self.OUT["nodeData/spheroidVelocity"]))
        # Determine spheroid-to-total mass ratio
        baryonicSpheroidMass = np.copy(np.array(self.OUT["nodeData/spheroidMassStellar"])) + \
            np.copy(np.array(self.OUT["nodeData/spheroidMassGas"]))
        baryonicDiskMass = np.copy(np.array(self.OUT["nodeData/diskMassStellar"])) + \
            np.copy(np.array(self.OUT["nodeData/diskMassGas"]))
        baryonicSpheroidToTotalRatio = baryonicSpheroidMass/(baryonicSpheroidMass+baryonicDiskMass)
        # Check if any disk-dominated galaxies in dataset and replace corresponding velocities
        diskDominated = baryonicSpheroidToTotalRatio<0.5
        if any(diskDominated):
            # Approximate disk velocity dispersion using combiantion of disk rotational velocity
            # and disk vertical velocity (computed as fraction of rotation velocity)
            diskVelocity = np.copy(np.array(self.OUT["nodeData/diskVelocity"]))
            inclination = getInclination(self.galacticusOBJ,float(self.redshift))*Pi/180.0
            diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)            
            np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
        return approximateVelocityDispersion


    def getFWHM(self,lineName,lineWidth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract rest wavelength
        restWavelength = self.EmissionLines.getWavelength(lineName)
        # Compute FWHM        
        if fnmatch.fnmatch(lineWidth.lower(),"fixedwidth*"):            
            widthVelocity = float(lineWidth.replace("fixedWidth",""))
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
        luminosity = amplitude*np.exp(-((wavelengths-lineWavelength)**2)/(2.0*(sigma**2)))
        return luminosity


    def cauchy(self,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        gamma = FWHM/2.0
        # Compute amplitude for Gaussian
        amplitude = np.stack([lineLuminosity]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= Pi*gamma
        # Compute luminosity
        wavelengths = np.concatenate([self.sedWavelengths]*len(lineLuminosity)).reshape(-1,len(self.sedWavelengths))
        luminosity = amplitude*(gamma**2)/(((wavelengths-lineWavelength)**2)+gamma**2)
        return luminosity





class EmissionLineProfiles_copy(object):
    
    def __init__(self,galObj,frame,redshift,wavelengths,selectionMask=None,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store GalacticusHDF5 object
        self.galacticusOBJ = galObj
        # Store redshift
        self.redshift = redshift
        # Store frame
        self.frame = frame
        # Store wavelengths
        self.sedWavelengths = wavelengths
        # Crete emission lines object
        self.EmissionLines = GalacticusEmissionLines()        
        # Check for any emission lines inside wavelength range
        lines = np.array(self.EmissionLines.getLineNames())
        if self.frame == "rest":
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name) for name in lines])
        else:
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name,redshift=float(redshift)) for name in lines])
        emissionLinesInRange = np.logical_and(effectiveWavelengths>=self.sedWavelengths.min(),effectiveWavelengths<=self.sedWavelengths.max())
        self.linesInRange = lines[emissionLinesInRange]
        # Exit here if no lines in range
        if len(self.linesInRange) == 0:
            return
        # Store wavelengths the lines appear at
        self.wavelengthsInRange = effectiveWavelengths[emissionLinesInRange]
        # Store selection mask to apply to galaxies
        self.selectionMask = selectionMask
        # Store redshift output
        self.OUT = self.galacticusOBJ.selectOutput(float(redshift))
        # Create 2D array to store sum of line profiles 
        if self.selectionMask is None:
            ngals = len(np.array(self.OUT["nodeData/nodeIndex"]))
        else:
            ngals = len(np.array(self.OUT["nodeData/nodeIndex"])[self.selectionMask])
        self.profileSum = np.zeros((ngals,len(self.sedWavelengths)),dtype=np.float64)
        return

    
    def sumLineProfiles(self,MATCH,ignoreResolution=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sum profiles
        dummy = [self.addLine(lineName,MATCH,ignoreResolution=ignoreResolution) for lineName in self.linesInRange]
        # Return sum
        frequency = speedOfLight/np.stack([self.sedWavelengths]*self.profileSum.shape[0])*angstrom
        self.profileSum /= frequency
        return self.profileSum


    def addLine(self,lineName,MATCH,ignoreResolution=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset information
        component = MATCH.group(1)
        resolution = MATCH.group(2)
        frame = MATCH.group(3)
        lineInformation = MATCH.group(4)
        snr = MATCH.group(5)
        if snr is not None:
            snr = snr.replace(":snr","")
        redshift = MATCH.group(6)
        dust = MATCH.group(7)
        if dust is None:
            dust = ""
        option = MATCH.group(8)
        if option is None:
            option = ""
        # Extra line profile information
        INFO = re.search(r"^:([^_]+)_([^_]+)",lineInformation)
        profile = INFO.group(1)
        lineWidth = INFO.group(2)
        # Get dataset name
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+dust+option
        if lineDatasetName not in self.OUT["nodeData"].keys():
            print("WARNING! "+funcname+"(): Cannot locate '"+lineDatasetName+"' for inclusion in SED. Will be skipped.")
            return
        # Get observed wavelength of line 
        iline = np.argwhere(self.linesInRange==lineName)[0]
        lineWavelength = self.wavelengthsInRange[iline][0]
        # Extract line luminosity
        lineLuminosity = np.array(self.OUT["nodeData/"+lineDatasetName])
        if self.selectionMask is not None:
            lineLuminosity = lineLuminosity[self.selectionMask]
        np.place(lineLuminosity,lineLuminosity<0.0,0.0)
        # Compute FWHM (use fixed line width in km/s)
        FWHM = self.getFWHM(lineName,lineWidth)
        # Check if FWHM larger than resolution (if resolution specified)
        if not ignoreResolution:
            resolutionLimit = lineWavelength/float(resolution)
            FWHM = np.maximum(FWHM,resolutionLimit)
        # Compute line profile
        if fnmatch.fnmatch(profile.lower(),"gaus*"):
            self.profileSum += self.gaussian(lineWavelength,lineLuminosity,FWHM)
        elif fnmatch.fnmatch(profile.lower(),"cauchy*"):
            self.profileSum += self.cauchy(lineWavelength,lineLuminosity,FWHM)
        else:
            raise ValueError(funcname+"(): line profile must be Gaussian or Cauchy! Other profiles not yet implemented!")
        return 


    def getApproximateVelocityDispersion(self,scaleVelocityRatio=0.1):
        """
        
        diskVelocityDispersion = diskVelocity * SQRT(sin(inclination)**2+(scaleVelocityRatio*cos(inclination))**2)

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Approximate spheroid velocity dispersion using spheroid 'rotation velocity'
        approximateVelocityDispersion = np.copy(np.array(self.OUT["nodeData/spheroidVelocity"]))
        # Determine spheroid-to-total mass ratio
        baryonicSpheroidMass = np.copy(np.array(self.OUT["nodeData/spheroidMassStellar"])) + \
            np.copy(np.array(self.OUT["nodeData/spheroidMassGas"]))
        baryonicDiskMass = np.copy(np.array(self.OUT["nodeData/diskMassStellar"])) + \
            np.copy(np.array(self.OUT["nodeData/diskMassGas"]))
        baryonicSpheroidToTotalRatio = baryonicSpheroidMass/(baryonicSpheroidMass+baryonicDiskMass)
        # Check if any disk-dominated galaxies in dataset and replace corresponding velocities
        diskDominated = baryonicSpheroidToTotalRatio<0.5
        if any(diskDominated):
            # Approximate disk velocity dispersion using combiantion of disk rotational velocity
            # and disk vertical velocity (computed as fraction of rotation velocity)
            diskVelocity = np.copy(np.array(self.OUT["nodeData/diskVelocity"]))
            inclination = getInclination(self.galacticusOBJ,float(self.redshift))*Pi/180.0
            diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)            
            np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
        return approximateVelocityDispersion


    def getFWHM(self,lineName,lineWidth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract rest wavelength
        restWavelength = self.EmissionLines.getWavelength(lineName)
        # Compute FWHM        
        if fnmatch.fnmatch(lineWidth.lower(),"fixedwidth*"):            
            widthVelocity = float(lineWidth.replace("fixedWidth",""))
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
        luminosity = amplitude*np.exp(-((wavelengths-lineWavelength)**2)/(2.0*(sigma**2)))
        return luminosity


    def cauchy(self,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        gamma = FWHM/2.0
        # Compute amplitude for Gaussian
        amplitude = np.stack([lineLuminosity]*len(self.sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= Pi*gamma
        # Compute luminosity
        wavelengths = np.concatenate([self.sedWavelengths]*len(lineLuminosity)).reshape(-1,len(self.sedWavelengths))
        luminosity = amplitude*(gamma**2)/(((wavelengths-lineWavelength)**2)+gamma**2)
        return luminosity
