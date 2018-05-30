#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from .io import GalacticusHDF5
from .galaxyProperties import DatasetClass
from .EmissionLines import GalacticusEmissionLine
from .StellarLuminosities import StellarLuminosities,parseStellarLuminosity
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



class topHatArrayClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,filterNames=None,luminosities=None,wavelengths=None):
        super(topHatArrayClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.filterNames = filterNames
        self.luminosities = luminosities
        self.wavelengths = wavelengths
        return
    
    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.filterNames = None
        self.luminosities = None
        self.wavelengths = None
        return

    
class sedTopHatArray(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        """
        sedTopHatArray: Class to store information for array of SED top hat filters included in Galacticus HDF5 output.

        USAGE:  THA = sedTophatArray(galHDF5Obj,z,[verbose])
        
            INPUTS:
            galHDF5Obj : GalacticusHDF5 object
            verbose       : Print additional information (T/F, default=F)

            OUTPUTS:
            THA           : sedTopHatArray object.

        """
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.STELLAR = StellarLuminosities(self.galHDF5Obj)
        return

    
    def createSEDTopHatArrayClass(self,z,frame,component='total',dust=None,recent=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initialize class
        TOPHATS = topHatArrayClass()
        # Create dataset name
        zString = self.galHDF5Obj.getRedshiftString(z)
        name = component.lower()+"LuminositiesStellar:sedTopHat_9999_9999:"+frame.lower()+":"+zString
        if dust is not None:
            if not dust.startswith(":"):
                dust = ":" + dust
            name = name + dust
        if recent:
            name = name + ":recent"
        TOPHATS.datasetName = parseStellarLuminosity(name)
        # Identify HDF5 output
        TOPHATS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            TOPHATS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            TOPHATS.redshift = np.ones(ngals,dtype=float)*redshift
        return TOPHATS


    def findTopHatFilters(self,TOPHATS):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        search = TOPHATS.datasetName.group(0).replace("sedTopHat_9999_9999","sedTopHat_*_*")
        z = float(TOPHATS.datasetName.group('redshift'))
        if fnmatch.fnmatch(TOPHATS.datasetName.group('component'),"total"):
            search = search.replace("total","disk")
        topHats = fnmatch.filter(self.STELLAR.availableLuminosities(z),search)
        topHats = np.array([name.replace("disk","total") for name in topHats])
        # Sort by central wavelength
        filterNames = np.unique([name.split(":")[1] for name in topHats])
        centralWavelength = np.array([float(name.split("_")[1]) for name in filterNames])
        isort = np.argsort(centralWavelength)
        TOPHATS.filterNames = np.copy(topHats[isort])
        TOPHATS.wavelengths = np.copy(centralWavelength[isort])        
        return TOPHATS


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


    def getAvailableWavelengths(self,z,frame,component="total",dust=None,recent=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Build top hat array class
        TOPHATS = self.createSEDTopHatArrayClass(z,frame,component=component,dust=dust,recent=recent)
        TOPHATS = self.findTopHatFilters(TOPHATS)
        return TOPHATS.wavelengths


    def setLuminosities(self,z,frame,component='total',dust=None,recent=False,selectionMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Build top hat array class
        TOPHATS = self.createSEDTopHatArrayClass(z,frame,component=component,dust=dust,recent=recent)
        TOPHATS = self.findTopHatFilters(TOPHATS)
        # Set selection mask if not already set
        if selectionMask is None:
            selectionMask = np.ones(len(TOPHATS.redshift),dtype=bool)
        else:
            if selectionMask.shape != TOPHATS.redshift.shape:
                raise KeyError(funcname+"(): length of selection mask does not equal number of galaxies.")
        # Extract luminosities
        luminosities = [self.STELLAR.getLuminosity(name)[selectionMask] for name in TOPHATS.filterNames]
        TOPHATS.luminosities = np.stack(np.copy(luminosities),axis=1)
        del luminosities
        return TOPHATS

        
    def getLuminosities(self,z,frame,component='total',dust=None,recent=False,selectionMask=None):
        """
        sedTopHatArray.getLuminosities: Return luminosities for top hat filter set.

        USAGE:  luminosities = sedTopHatArray.getLuminosities(z,frame,[component],[dust],[recent],[selectionMask])

             INPUTS:
             z               : Redshift of snapshot to extract luminosities from.
             frame           : Frame of reference ('rest' or 'observed')
             component       : Component of galaxy ('disk', 'spheroid', or 'total') [Default='total'] 
             dust            : Dust information for filter name (e.g. ':dustAtlasClouds') [Default = None]
             recent          : Consider only recent star formation (True/False). [Default=False]
             selectionMask   : Logical mask to select subset of galaxies.

             OUTPUTS:
             luminosities    : 2D array of luminosities of size (nfilters x ngalaxies).
        
        """
        TOPHATS = self.setLuminosities(z,frame,component=component,dust=dust,recent=recent,selectionMask=selectionMask)
        return TOPHATS.luminosities
    



class SEDClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,wavelength=None,continuum=None,\
                     emissionLines=None,sed=None,mask=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(SEDClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.wavelength = wavelength
        self.continuum = continuum
        self.emissionLines = emissionLines
        self.sed = sed
        self.mask = mask
        return

    def reset(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.wavelength = None
        self.continuum = None
        self.emissionLines = None
        self.sed = None
        self.mask = None
        return

    def updateWavelengthInterval(self,wavelengthInterval):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        minWavelength = self.wavelengths.min()
        maxWavelength = self.wavelengths.max()    
        nbins = len(np.arange(minWavelength,maxWavelength,wavelengthInterval))
        newWavelengths = np.linspace(minWavelength,maxWavelength,nbins+1)
        self.sed = np.copy(interpolateSED(np.copy(self.wavelengths),np.copy(self.sed),\
                                      newWavelengths,axis=1))
        self.wavelengths = np.copy(newWavelengths)
        return




def parseSED(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)SED:(?P<frame>[^:]+)"+\
        "(?P<emlines>:(?P<lineProfile>[^:_]+)_(?P<lineWidth>[^:]+))?"+\
        "(?P<snrString>:snr(?P<snr>[\d\.]+))?"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
        "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
    MATCH = re.search(searchString,datasetName)
    # Check dataset name corresponds to an SED
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class GalacticusSED(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.verbose = verbose  
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj
        # Initialise top hat array object
        self.TOPHATS = sedTopHatArray(self.galHDF5Obj)
        # Initialise emission lines object
        self.EmissionLines = EmissionLineProfiles(self.galHDF5Obj)
        return
            
    def createSEDClass(self,datasetName,wavelengths,galaxyMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        SED = SEDClass()
        SED.datasetName = parseSED(datasetName)
        # Identify HDF5 output
        SED.outputName = self.galHDF5Obj.nearestOutputName(float(SED.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(SED.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            SED.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(SED.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            SED.redshift = np.ones(ngals,dtype=float)*z
        # Store wavelengths
        SED.wavelength = wavelengths
        # Store mask and initialize SED array
        if galaxyMask is None:
            galaxyMask = np.ones(len(SED.redshift),dtype=bool)
        SED.mask = np.copy(galaxyMask)
        SED.sed = np.zeros((np.count_nonzero(SED.mask),len(SED.wavelength)),dtype=float)
        return SED

    def getAvailableWavelengths(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name   
        # Construct SED class to extract dataset name components
        SED = SEDClass()
        SED.datasetName = parseSED(datasetName)
        # Extract minimum and maximum wavelength
        z = float(SED.datasetName.group('redshift'))
        frame = SED.datasetName.group('frame')
        component = SED.datasetName.group('component')
        dust = SED.datasetName.group('dust')
        recent = SED.datasetName.group('recent') is not None
        wavelengths = self.TOPHATS.getAvailableWavelengths(z,frame,component=component,\
                                                               dust=dust,recent=recent)
        return wavelengths

    def addContinuumNoise(self,SED):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if SED.datasetName.group('snr') is None:
            return SED
        # Get S/N ratio to use
        SNR = float(SED.datasetName.group('snr'))
        # Get Poisson error on count rate of photons
        energy = speedOfLight*plancksConstant/np.stack([SED.wavelength]*SED.continuum.shape[0])*angstrom
        counts = SED.continuum*luminosityAB/energy
        # Perturb counts and convert back to luminosities
        counts = norm.rvs(loc=counts,scale=counts/SNR)
        SED.continuum = np.copy(counts*energy/luminosityAB)
        return SED

    def setContinuumSED(self,SED,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set luminosities for top hat filter array
        z = float(SED.datasetName.group('redshift'))
        frame = SED.datasetName.group('frame')
        component = SED.datasetName.group('component')
        dust = SED.datasetName.group('dust')
        recent = SED.datasetName.group('recent') is not None
        TOPHATS = self.TOPHATS.setLuminosities(z,frame,component=component,dust=None,\
                                                   recent=False,selectionMask=SED.mask)
        # Interpolate to get continuum SED
        SED.continuum = interpolateSED(TOPHATS.wavelengths,TOPHATS.luminosities,\
                                           SED.wavelength,axis=1,**kwargs)
        # Add in continuum noise
        SED = self.addContinuumNoise(SED)
        return SED

    def setEmissionLineProfiles(self,SED):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if SED.datasetName.group('emlines') is not None:
            SED.emissionLines = self.EmissionLines.getLineProfiles(SED.datasetName.group(0),SED.wavelength,\
                                                                      galaxyMask=SED.mask)
        else:
            SED.emissionLines = np.zeros((np.count_nonzero(SED.mask),len(SED.wavelength)),dtype=float)
        return SED

    def ergPerSecond(self,sed,zeroCorrection=1.0e-50):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        sed = np.log10(sed+zeroCorrection)
        sed += np.log10(luminosityAB)
        sed -= np.log10(erg)
        sed = 10.0**sed
        return sed

    def convertToMicroJanskies(self,z,sed):
        sed = self.ergPerSecond(sed)
        comDistance = self.galHDF5Obj.cosmology.comoving_distance(z)*megaParsec/centi
        comDistance = np.repeat(comDistance,sed.shape[1]).reshape(sed.shape)
        sed /= 4.0*Pi*comDistance**2
        sed /= jansky
        sed *= 1.0e6
        return sed

    def computeTotalSED(self,SED):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        SED.continuum = self.convertToMicroJanskies(SED.redshift[SED.mask],SED.continuum)
        SED.emissionLines = self.convertToMicroJanskies(SED.redshift[SED.mask],SED.emissionLines)
        SED.sed = SED.continuum + SED.emissionLines
        return SED
    
    def buildSED(self,datasetName,wavelengths,galaxyMask=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create class to store SED information
        SED = self.createSEDClass(datasetName,wavelengths,galaxyMask=galaxyMask)
        # Get continuum luminosity
        SED = self.setContinuumSED(SED,**kwargs)
        # Introduce emission lines
        SED = self.setEmissionLineProfiles(SED)
        # Compute the total SED
        SED = self.computeTotalSED(SED)
        return SED
        
    def getSED(self,datasetName,wavelengths,galaxyMask=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        SED = self.buildSED(datasetName,wavelengths,galaxyMask=galaxyyMask,**kwargs)        
        return SED.wavelength,SED.sed



class LineProfilesClass(DatasetClass):

    def __init__(self,datasetName=None,redshift=None,outputName=None,wavelengths=None,luminosities=None,mask=None):
        super(LineProfilesClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.wavelengths = wavelengths
        self.luminosities = luminosities
        self.mask = mask
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.wavelengths = None
        self.luminosities = None
        self.mask = None
        return



class EmissionLineProfiles(object):
    
    def __init__(self,galHDF5Obj,verbose=False,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj        
        # Create emission lines object
        self.EmissionLines = GalacticusEmissionLine(self.galHDF5Obj,**kwargs)                
        return

    def createLineProfilesClass(self,datasetName,wavelengths,galaxyMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LINES = LineProfilesClass()
        LINES.datasetName = parseSED(datasetName)
        # Identify HDF5 output
        LINES.outputName = self.galHDF5Obj.nearestOutputName(float(LINES.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(LINES.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT["nodeData"].keys():
            LINES.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(LINES.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            LINES.redshift = np.ones(ngals,dtype=float)*z
        # Store wavelengths
        LINES.wavelengths = np.copy(wavelengths)
        # Store mask and initialize SED array
        if galaxyMask is None:
            galaxyMask = np.ones(len(LINES.redshift),dtype=bool)
        LINES.mask = np.copy(galaxyMask)
        LINES.luminosities = np.zeros((np.count_nonzero(LINES.mask),len(LINES.wavelengths)),dtype=float)
        return LINES

    
    def setLineProfiles(self,datasetName,wavelengths,galaxyMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LINES = self.createLineProfilesClass(datasetName,wavelengths,galaxyMask=galaxyMask)
        LINES = self.sumLineProfiles(LINES)
        return LINES

    
    def getLineProfiles(self,datasetName,wavelengths,galaxyMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        LINES = self.setLineProfiles(datasetName,wavelengths,galaxyMask=galaxyMask)
        return LINES.luminosities

    
    def sumLineProfiles(self,LINES):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sum profiles
        dummy = [self.addLine(LINES,lineName) for lineName in self.EmissionLines.getLineNames()]
        # Return sum
        frequency = speedOfLight/np.stack([LINES.wavelengths]*LINES.luminosities.shape[0])*angstrom
        LINES.luminosities /= frequency
        return LINES


    def addLine(self,LINES,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset information
        component = LINES.datasetName.group('component')
        frame = LINES.datasetName.group('frame')
        lineProfile = LINES.datasetName.group('lineProfile')
        lineWidth = LINES.datasetName.group('lineWidth')
        redshift = LINES.datasetName.group('redshift')
        dust = LINES.datasetName.group('dust')
        if not dust:
            dust = ""
        recent = LINES.datasetName.group('recent')
        if not recent:
            recent = ""
        # Get observed wavelength of line 
        lineWavelength = self.EmissionLines.getWavelength(lineName)
        if fnmatch.fnmatch(frame,"observed"):
            lineWavelength *= (1.0+LINES.redshift)
        else:
            lineWavelength = np.ones_like(LINES.redshift)*lineWavelength
        lineWavelength = lineWavelength[LINES.mask]
        # Get line luminosity
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+recent+dust
        lineLuminosity = self.EmissionLines.getLineLuminosity(lineDatasetName)[LINES.mask]
        np.place(lineLuminosity,lineLuminosity<0.0,0.0)
        # Compute FWHM (use fixed line width in km/s)
        FWHM = self.getFWHM(LINES,lineName,lineWidth)
        # Compute line profile
        if fnmatch.fnmatch(lineProfile.lower(),"gaus*"):
            LINES.luminosities += self.gaussian(LINES.wavelengths,lineWavelength,lineLuminosity,FWHM)
        elif fnmatch.fnmatch(lineProfile.lower(),"voigt*"):
            LINES.luminosities += self.cauchy(LINES.wavelengths,lineWavelength,lineLuminosity,FWHM)
        else:
            raise ValueError(funcname+"(): line profile must be Gaussian or Voigt! Other profiles not yet implemented!")
        return LINES

    
    def getBaryonicBulgeToTotalRatio(self,z,emptyHalos=999.9):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Determine spheroid-to-total mass ratio
        baryonicSpheroidMass = np.copy(np.array(HDF5OUT["nodeData/spheroidMassStellar"])) + \
            np.copy(np.array(HDF5OUT["nodeData/spheroidMassGas"])) 
        baryonicDiskMass = np.copy(np.array(HDF5OUT["nodeData/diskMassStellar"])) + \
            np.copy(np.array(HDF5OUT["nodeData/diskMassGas"])) 
        totalBaryonicMass = baryonicSpheroidMass + baryonicDiskMass
        mask = totalBaryonicMass == 0.0
        np.place(totalBaryonicMass,mask,1.0)
        np.place(baryonicSpheroidMass,mask,emptyHalos)
        return baryonicSpheroidMass/totalBaryonicMass
        


    def getApproximateVelocityDispersion(self,z,scaleVelocityRatio=0.1,minVelocityDipserion=0.001):
        """
        diskVelocityDispersion = diskVelocity * SQRT(sin(inclination)**2+(scaleVelocityRatio*cos(inclination))**2)
        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Approximate spheroid velocity dispersion using spheroid 'rotation velocity'
        approximateVelocityDispersion = np.copy(np.array(HDF5OUT["nodeData/spheroidVelocity"]))
        # Determine spheroid-to-total mass ratio
        emptyHalos = 999.9
        baryonicSpheroidToTotalRatio = self.getBaryonicBulgeToTotalRatio(z,emptyHalos=emptyHalos)
        # Fort any empty halos not removed by mask, set velocity dispersion to specified minium value
        mask = baryonicSpheroidToTotalRatio == emptyHalos
        np.place(approximateVelocityDispersion,mask,minVelocityDipserion)
        # Check if any disk-dominated galaxies in dataset and replace corresponding velocities
        diskDominated = baryonicSpheroidToTotalRatio<0.5
        if any(diskDominated):
            # Approximate disk velocity dispersion using combiantion of disk rotational velocity
            # and disk vertical velocity (computed as fraction of rotation velocity)
            diskVelocity = np.copy(np.array(HDF5OUT["nodeData/diskVelocity"]))            
            inclination = getInclination(self.galHDF5Obj,z)
            diskVelocity *= np.sqrt(np.sin(inclination)**2+(scaleVelocityRatio*np.cos(inclination))**2)            
            np.place(approximateVelocityDispersion,diskDominated,diskVelocity[diskDominated])
        return approximateVelocityDispersion


    def getFWHM(self,LINES,lineName,lineWidth):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract rest wavelength
        restWavelength = self.EmissionLines.getWavelength(lineName)
        # Compute FWHM        
        if fnmatch.fnmatch(lineWidth.lower(),"fixedwidth*"):                        
            widthVelocity = np.ones_like(LINES.luminosities)*float(lineWidth.replace("fixedWidth",""))
        elif lineWidth.lower() == "velocityapprox":
            widthVelocity = self.getApproximateVelocityDispersion(float(LINES.datasetName.group('redshift')),scaleVelocityRatio=0.1)
            widthVelocity = np.stack([widthVelocity]*len(LINES.wavelengths),axis=1).reshape(len(widthVelocity),-1)
        else:
            raise ValueError(funcname+"(): line width method must be 'fixed'! Other methods not yet implemented!")
        c = speedOfLight/kilo
        FWHM = restWavelength*(widthVelocity/c)    
        return FWHM
    

    def gaussian(self,sedWavelengths,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        sigma = FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))
        # Compute amplitude for Gaussian
        amplitude = np.stack([lineLuminosity]*len(sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= (sigma*np.sqrt(2.0*Pi))        
        # Compute luminosity
        wavelengths = np.concatenate([sedWavelengths]*len(lineLuminosity)).reshape(-1,len(sedWavelengths))
        lineWavelengths = np.stack([lineWavelength]*len(sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        luminosity = amplitude*np.exp(-((wavelengths-lineWavelengths)**2)/(2.0*(sigma**2)))
        return luminosity


    def voigt(self,sedWavelengths,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        gamma = FWHM/2.0
        # Compute amplitude for Voigt profile
        amplitude = np.stack([lineLuminosity]*len(sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                
        amplitude /= Pi*gamma
        # Compute luminosity
        wavelengths = np.concatenate([sedWavelengths]*len(lineLuminosity)).reshape(-1,len(sedWavelengths))
        lineWavelengths = np.stack([lineWavelength]*len(sedWavelengths),axis=1).reshape(len(lineLuminosity),-1)                    
        luminosity = amplitude*(gamma**2)/(((wavelengths-lineWavelengths)**2)+gamma**2)
        return luminosity



