#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import xml.etree.ElementTree as ET
import pkg_resources
from scipy.interpolate import RegularGridInterpolator

from .utils import DustProperties
from ..io import GalacticusHDF5
from ..EmissionLines import GalacticusEmissionLines
from ..Inclination import getInclination
from ..GalacticusErrors import ParseError
from ..Filters import GalacticusFilters
from ..config import *
from ..utils.progress import Progress
from ..constants import luminosityAB,luminositySolar
from ..constants import Pi,Parsec,massAtomic,massSolar,massFractionHydrogen


def loadDiskAttenuation(component,verbose=False):               
    data = {}
    data["attenuation"] = None       
    # Loop over inclinations
    inclinations = component.findall("inclination")            
    data["inclination"] = np.zeros(len(inclinations),dtype=float)    
    inclinationCount = data["inclination"].size
    PROG = Progress(inclinationCount)
    for i,inclination in enumerate(inclinations):      
        data["inclination"][i] = float(inclination.find("angle").text)
        # Loop over optical depths
        opticalDepths = inclination.findall("opticalDepth")
        if i == 0:
            data["opticalDepth"] = np.zeros(len(opticalDepths),dtype=float)
        opticalDepthCount = data["opticalDepth"].size
        for j,opticalDepth in enumerate(opticalDepths):
            if i == 0:
                data["opticalDepth"][j] = float(opticalDepth.find("tau").text)
            # Read attenuations for this optical depth
            atten = np.copy([ float(att.text) for att in opticalDepth.iter("attenuation")])            
            # Create array to store attenuations
            if data["attenuation"] is None:
                data["attenuation"] = np.zeros((inclinationCount,opticalDepthCount,atten.size))
            data["attenuation"][i,j,:] = np.copy(atten)
            del atten
        PROG.increment()
        if verbose:
            PROG.print_status_line()
    return data


def loadSpheroidAttenuation(component,verbose=False):               
    data = {}
    data["attenuation"] = None       
    # Loop over bulge sizes
    sizes = component.findall("bulgeSize")            
    data["size"] = np.zeros(len(sizes),dtype=float)    
    sizeCount = data["size"].size
    PROG = Progress(sizeCount)
    for i,bulgeSize in enumerate(sizes):      
        data["size"][i] = float(bulgeSize.find("size").text)
        # Loop over inclinations        
        inclinations = bulgeSize.findall("inclination")            
        if i == 0:
            data["inclination"] = np.zeros(len(inclinations),dtype=float)    
        inclinationCount = data["inclination"].size
        for j,inclination in enumerate(inclinations):      
            data["inclination"][j] = float(inclination.find("angle").text)
            # Loop over optical depths
            opticalDepths = inclination.findall("opticalDepth")
            if i == 0:
                data["opticalDepth"] = np.zeros(len(opticalDepths),dtype=float)
            opticalDepthCount = data["opticalDepth"].size
            for k,opticalDepth in enumerate(opticalDepths):
                if i == 0:
                    data["opticalDepth"][k] = float(opticalDepth.find("tau").text)
                # Read attenuations for this optical depth
                atten = np.copy([ float(att.text) for att in opticalDepth.iter("attenuation")])            
                # Create array to store attenuations
                if data["attenuation"] is None:
                    data["attenuation"] = np.zeros((sizeCount,inclinationCount,opticalDepthCount,atten.size))
                data["attenuation"][i,j,k,:] = np.copy(atten)
                del atten
        PROG.increment()
        if verbose:
            PROG.print_status_line()
    return data



class dustAtlasTable(object):
    
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        self.dustFile = pkg_resources.resource_filename(__name__,"../data/dust/Ferrara2000/attenuations_MilkyWay_dustHeightRatio1.0.xml")
        if not os.path.exists(self.dustFile):
            raise IOError(classname+"(): Cannot find Ferrara et al. (1999) dust atlas file!")
        if self.verbose:
            print(classname+"(): Loading Ferrara et al. (1999) dust atlas file...")                    
        # Construct map of XML parents/descendents
        self.dustData = ET.parse(self.dustFile)
        self.dustRoot = self.dustData.getroot()
        self.dustMap = {c.tag:p for p in self.dustRoot.iter() for c in p}        
        # Load wavelengths
        if self.verbose:
            print(classname+"(): Extracting wavelengths...")
        self.wavelengths = np.copy([ float(lam.text) for lam in self.dustMap["wavelengths"].iter("lambda")])        
        # Create varaibles for interpolators
        self.spheroidAttenuationGrid = None
        self.diskAttenuationGrid = None
        self.spheroidInterpolator = None
        self.diskInterpolator = None
        return

    def loadSpheroidAttenuationGrid(self,component):
        data = {}
        data["attenuation"] = None       
        # Loop over bulge sizes
        sizes = component.findall("bulgeSize")            
        data["size"] = np.zeros(len(sizes),dtype=float)    
        sizeCount = data["size"].size
        PROG = Progress(sizeCount)
        for i,bulgeSize in enumerate(sizes):      
            data["size"][i] = float(bulgeSize.find("size").text)
            # Loop over inclinations        
            inclinations = bulgeSize.findall("inclination")            
            if i == 0:
                data["inclination"] = np.zeros(len(inclinations),dtype=float)    
            inclinationCount = data["inclination"].size
            for j,inclination in enumerate(inclinations):      
                data["inclination"][j] = float(inclination.find("angle").text)
                # Loop over optical depths
                opticalDepths = inclination.findall("opticalDepth")
                if i == 0:
                    data["opticalDepth"] = np.zeros(len(opticalDepths),dtype=float)
                opticalDepthCount = data["opticalDepth"].size
                for k,opticalDepth in enumerate(opticalDepths):
                    if i == 0:
                        data["opticalDepth"][k] = float(opticalDepth.find("tau").text)
                    # Read attenuations for this optical depth
                    atten = np.copy([ float(att.text) for att in opticalDepth.iter("attenuation")])            
                    # Create array to store attenuations
                    if data["attenuation"] is None:
                        data["attenuation"] = np.zeros((sizeCount,inclinationCount,opticalDepthCount,atten.size))
                    data["attenuation"][i,j,k,:] = np.copy(atten)
                    del atten
            PROG.increment()
            if self.verbose:
                PROG.print_status_line()
        self.spheroidAttenuationGrid = data
        return

    def loadDiskAttenuationGrid(self,component):               
        data = {}
        data["attenuation"] = None       
        # Loop over inclinations
        inclinations = component.findall("inclination")            
        data["inclination"] = np.zeros(len(inclinations),dtype=float)    
        inclinationCount = data["inclination"].size
        PROG = Progress(inclinationCount)
        for i,inclination in enumerate(inclinations):      
            data["inclination"][i] = float(inclination.find("angle").text)
            # Loop over optical depths
            opticalDepths = inclination.findall("opticalDepth")
            if i == 0:
                data["opticalDepth"] = np.zeros(len(opticalDepths),dtype=float)
            opticalDepthCount = data["opticalDepth"].size
            for j,opticalDepth in enumerate(opticalDepths):
                if i == 0:
                    data["opticalDepth"][j] = float(opticalDepth.find("tau").text)
                # Read attenuations for this optical depth
                atten = np.copy([ float(att.text) for att in opticalDepth.iter("attenuation")])            
                # Create array to store attenuations
                if data["attenuation"] is None:
                    data["attenuation"] = np.zeros((inclinationCount,opticalDepthCount,atten.size))
                data["attenuation"][i,j,:] = np.copy(atten)
                del atten
            PROG.increment()
        if self.verbose:
            PROG.print_status_line()
        self.diskAttenuationGrid = data            
        return 

    def buildSpheroidInterpolator(self,component,interpolateBoundsError=False,interpolateFillValue=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.spheroidAttenuationGrid is None:
            if self.verbose:
                print(funcname+"(): Loading spheroid attenuations...")
            self.loadSpheroidAttenuationGrid(component)
        if self.verbose:
            print(funcname+"(): Building spheroid interpolator...")            
        axes = (self.spheroidAttenuationGrid["size"],self.spheroidAttenuationGrid["inclination"],\
                    self.spheroidAttenuationGrid["opticalDepth"],self.wavelengths)                
        self.spheroidInterpolator = RegularGridInterpolator(axes,self.spheroidAttenuationGrid["attenuation"],\
                                                                bounds_error=interpolateBoundsError,\
                                                                fill_value=interpolateFillValue)        
        return

    
    def buildDiskInterpolator(self,component,interpolateBoundsError=False,interpolateFillValue=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.diskAttenuationGrid is None:
            if self.verbose:
                print(funcname+"(): Loading disk attenuations...")
            self.loadDiskAttenuationGrid(component)
        if self.verbose:
            print(funcname+"(): Building disk interpolator...")            
        
        axes = (self.diskAttenuationGrid["inclination"],self.diskAttenuationGrid["opticalDepth"],self.wavelengths)
        self.diskInterpolator = RegularGridInterpolator(axes,self.diskAttenuationGrid["attenuation"],\
                                                            bounds_error=interpolateBoundsError,\
                                                            fill_value=interpolateFillValue)                                                                        
        return

    def buildInterpolator(self,component,interpolateBoundsError=False,interpolateFillValue=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component == "spheroid":
            component = "bulge"
        for comp in self.dustData.iter("components"):
            if comp.find("name").text == component:                
                if component in "bulge spheroid".split():
                    self.buildSpheroidInterpolator(comp,interpolateBoundsError=interpolateBoundsError,\
                                                       interpolateFillValue=interpolateFillValue)
                if component == "disk":
                    self.buildDiskInterpolator(comp,interpolateBoundsError=interpolateBoundsError,\
                                                   interpolateFillValue=interpolateFillValue)

        return

    def interpolateDustTable(self,component,wavelength,inclination,opticalDepth,bulgeSize=None,\
                                 interpolateBoundsError=False,interpolateFillValue=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if component.lower() == "disk":
            if self.diskInterpolator is None:
                self.buildInterpolator("disk",interpolateBoundsError=interpolateBoundsError,\
                                           interpolateFillValue=interpolateFillValue)
            attenuation = self.diskInterpolator(zip(inclination,opticalDepth,wavelength))
        elif component.lower() == "spheroid":
            if bulgeSize is None:
                raise TypeError(funcname+"(): For spheroid dust attenuation, need to provide bulge sizes!")
            if self.spheroidInterpolator is None:
                self.buildInterpolator("bulge",interpolateBoundsError=interpolateBoundsError,\
                                           interpolateFillValue=interpolateFillValue)
            attenuation = self.spheroidInterpolator(zip(bulgeSize,inclination,opticalDepth,wavelength))
        else:
            raise ValueError(funcname+"(): Value for 'component' not recognised! Must be either 'spheroid' or 'disk'!")
        np.place(attenuation,attenuation>1.0,1.0)
        np.place(attenuation,attenuation<0.0,0.0)
        return attenuation





class dustAtlasBase(DustProperties):


    def __init__(self,verbose=False,extrapolateInSize=True,extrapolateInTau=True,\
                     opticalDepthISMFactor=0.0,opticalDepthCloudsFactor=0.0,\
                     wavelengthZeroPoint = None,wavelengthExponent=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose        
        # Initialise DustProperties class
        super(dustAtlasBase, self).__init__()
        # Create dust table object
        self.dustTable = dustAtlasTable(verbose=self.verbose)
        # Set options whether to extrapolate for various properties
        self.extrapolateInSize = extrapolateInSize
        self.extrapolateInTau = extrapolateInTau
        # Set optical depth parameters for Charlot & Fall (2000) model
        self.opticalDepthISMFactor = opticalDepthISMFactor
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        return

    def setWavelengthParameters(self,wavelengthExponent=None,wavelengthZeroPoint=None):
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        return

    def setOpticalDepthFactors(self,opticalDepthISMFactor=0.0,opticalDepthCloudsFactor=0.0):
        self.opticalISMCloudsFactor = opticalDepthISMFactor
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        return

    def getBulgeSizes(self,component,spheroidMassDistribution,spheroidRadius,diskRadius):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): 'component' must be 'spheroid' or 'disk'!")
        if spheroidMassDistribution not in ["hernquist","sersic"]:
            raise ValueError(funcname+"(): 'spheroidMassDistribution' must be 'hernquist' or 'sersic'!")
        np.place(diskRadius,diskRadius<=0.0,1.0)
        # Component bulge sizes
        if component.lower() == "spheroid":
            if fnmatch.fnmatch(spheroidMassDistribution.lower(),"hernquist"):
                sizes = (1.0+np.sqrt(2.0))*spheroidRadius/diskRadius
            elif fnmatch.fnmatch(spheroidMassDistribution.lower(),"sersic"):
                sizes = spheroidRadius/diskRadius
            else:
                pass
            if not self.extrapolateInSize:
                sizeMinimum = self.spheroidAttenuation["size"].min()
                sizeMaximum = self.spheroidAttenuation["size"].max()
                np.place(sizes,sizes<sizeMinimum,sizeMinimum)
                np.place(sizes,sizes>sizeMaximum,sizeMaximum)
        else:
            sizes = None
        return sizes

    def getCloudGasMetalsSurfaceDensity(self,gasMass,gasMetalMass,scaleLength,mcloud=1.0e6,rcloud=16.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        np.place(gasMetalMass,gasMass==0.0,0.0)
        np.place(gasMass,gasMass==0.0,1.0)
        metallicity = gasMetalMass/gasMass
        # Compute surface density
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = metallicity*mcloud/(2.0*Pi*(mega*rcloud)**2)
        np.place(gasMetalsSurfaceDensityCentral,np.isnan(scaleLength),0.0)
        return gasMetalsSurfaceDensityCentral
    
    def getOpticalDepthCentral(self,gasMass,gasMetalMass,scaleLength,mcloud=1.0e6,rcloud=16.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        gasMetalsSurfaceDensityCentral = self.getCloudGasMetalsSurfaceDensity(gasMass,gasMetalMass,scaleLength,mcloud=mcloud,rcloud=rcloud)
        opticalDepthCentral = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)
        if not self.extrapolateInTau:
            if fnmatch.fnmatch(component,"spheroid"):
                tauMinimum = self.spheroidAttenuation["opticalDepth"].min()
                tauMaximum = self.spheroidAttenuation["opticalDepth"].max()
            elif fnmatch.fnmatch(component,"disk"):
                tauMinimum = self.diskAttenuation["opticalDepth"].min()
                tauMaximum = self.diskAttenuation["opticalDepth"].max()
            else:
                raise ValueError(funcname+"(): Value for 'component' not recognised -- must be 'spheroid' or 'disk'!")
            np.place(opticalDepthCentral,opticalDepthCentral<tauMinimum,tauMinimum)
            np.place(opticalDepthCentral,opticalDepthCentral>tauMaximum,tauMaximum)
        return opticalDepthCentral

    def computeOpticalDepthISM(self,gasMetalMass,scaleLength,effectiveWavelength,opticalDepthISMFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if opticalDepthISMFactor is None:
            opticalDepthISMFactor = self.opticalDepthISMFactor
        # Compute gas metals central surface density in M_Solar/pc^2
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepth = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = opticalDepthISMFactor*opticalDepth
        return opticalDepthISM

    def computeOpticalDepthClouds(self,gasMass,gasMetalMass,effectiveWavelength,opticalDepthCloudsFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if opticalDepthCloudsFactor is None:
            opticalDepthCloudsFactor = self.opticalDepthCloudsFactor
        # Compute gas metallicity
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepthClouds = opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        return opticalDepthClouds

    def computeAttenuationISM(self,component,effectiveWavelength,inclination,spheroidMassDistribution,\
                                  spheroidRadius,diskRadius,gasMass,gasMetalMass,scaleLength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        sizes = self.getBulgeSizes(component,spheroidMassDistribution,np.copy(spheroidRadius),np.copy(diskRadius))
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        opticalDepthCentral = self.getOpticalDepthCentral(np.copy(gasMass),np.copy(gasMetalMass),np.copy(scaleLength),mcloud=1.0e6,rcloud=16.0e-6)
        wavelengths = np.ones_like(inclination)*effectiveWavelength
        attenuationISM = self.dustTable.interpolateDustTable(component,wavelengths,inclination,opticalDepthCentral,bulgeSize=sizes)
        attenuationISM *= np.exp(-self.opticalDepthISMFactor)
        np.place(attenuationISM,diskRadius<=0.0,1.0)
        return attenuationISM



class dustAtlas(dustAtlasBase):
    
    def __init__(self,galHDF5Obj,verbose=False,extrapolateInSize=True,extrapolateInTau=True,\
                     opticalDepthISMFactor=0.0,opticalDepthCloudsFactor=0.0,\
                     wavelengthZeroPoint = None,wavelengthExponent=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise base class class
        super(dustAtlas,self).__init__(verbose=verbose,extrapolateInSize=extrapolateInSize,extrapolateInTau=extrapolateInTau,\
                                           opticalDepthISMFactor=opticalDepthISMFactor,opticalDepthCloudsFactor=opticalDepthCloudsFactor,\
                                           wavelengthZeroPoint=wavelengthZeroPoint,wavelengthExponent=wavelengthExponent)
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        # Initialise variables to store attenuation information
        self.opticalDepthISM = None
        self.opticalDepthClouds = None
        # Initialise variables to store attenuated luminosity
        self.attenuatedLuminosity = None
        return

    def resetHDF5Output(self):
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        return
    
    def reset(self):
        self.resetAttenuationInformation()
        return


    def resetAttenuationInformation(self):
        self.opticalDepthISM = None
        self.opticalDepthClouds = None
        self.attenuatedLuminosity = None
        return

    def setHDF5Output(self,z):
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
            searchString = "^(?P<component>disk|spheroid)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<filterName>:[^:]+)?(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustAtlas(?P<clouds>Clouds)?(?P<options>[^:]+)?)$"
        else:
            searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustAtlas(?P<clouds>Clouds)?(?P<options>[^:]+)?)$"
        self.datasetName = re.search(searchString,datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return

    def getEffectiveWavelength(self):
        redshift = None
        if self.datasetName.group('frame').replace(":","") == "observed":
            redshift = float(self.redshift)
        if 'lineName' in self.datasetName.re.groupindex.keys():
            name = self.datasetName.group('lineName')
            redshift = None
        else:
            name = self.datasetName.group('filterName')        
        effectiveWavelength = self.effectiveWavelength(name,redshift=redshift,verbose=self.verbose)
        return effectiveWavelength


    def setOpticalDepths(self,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Reset attenuation information
        self.resetAttenuationInformation()
        # Set component
        component = self.datasetName.group('component')
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        # Set effective wavelength
        effectiveWavelength = self.getEffectiveWavelength()
        #  Set inclination
        if "options" in self.datasetName.re.groupindex.keys():
            if self.datasetName.group('options') is None:
                inclination = np.copy(np.array(self.hdf5Output["nodeData/inclination"]))
            else:
                if fnmatch.fnmatch(self.datasetName.group('options').lower(),"faceon"):
                    inclination = np.zeros_like(np.array(self.hdf5Output["nodeData/diskRadius"]))
                else:
                    inclination = np.copy(np.array(self.hdf5Output["nodeData/inclination"]))
        else:
            inclination = np.copy(np.array(self.hdf5Output["nodeData/inclination"]))
        # Get radii and set scalelengths
        spheroidMassDistribution = self.galHDF5Obj.parameters["spheroidMassDistribution"].lower()
        spheroidRadius = np.copy(np.array(self.hdf5Output["nodeData/spheroidRadius"]))
        diskRadius = np.copy(np.array(self.hdf5Output["nodeData/diskRadius"]))
        scaleLength = np.copy(np.array(self.hdf5Output["nodeData/"+component+"Radius"]))
        # Get gas and metal masses
        gasMass = np.copy(np.array(self.hdf5Output["nodeData/"+component+"MassGas"]))
        gasMetalMass = np.copy(np.array(self.hdf5Output["nodeData/"+component+"AbundancesGasMetals"]))
        # Compute attenuation for ISM
        attenuationISM = self.computeAttenuationISM(component,effectiveWavelength,inclination,spheroidMassDistribution,\
                                                        spheroidRadius,diskRadius,gasMass,gasMetalMass,scaleLength)
        np.place(attenuationISM,diskRadius<=0.0,1.0)
        np.place(attenuationISM,attenuationISM==0.0,1.0e-20)
        self.opticalDepthISM = -np.log(attenuationISM)
        # Compute attenuation for molecular clouds
        if "clouds" in list(self.datasetName.groups()):
            self.opticalDepthClouds = self.computeOpticalDepthClouds(np.copy(gasMass),np.copy(gasMetalMass),effectiveWavelength)
        else:
            self.opticalDepthClouds = 0.0
        return

    def applyAttenuation(self,luminosity,recentLuminosity=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if recentLuminosity is None:
            recentLuminosity = np.zeros_like(luminosity)
        attenuationISM = np.exp(-self.opticalDepthISM)
        attenuationClouds = np.exp(-self.opticalDepthClouds)
        result = ((luminosity-recentLuminosity) + recentLuminosity*attenuationClouds)*attenuationISM
        return result
    
    def setAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set datasetName
        self.setDatasetName(datasetName)
        # Check HDF5 snapshot specified
        if z is not None:
            self.setHDF5Output(z)
        else:
            if self.hdf5Output is None:
                z = self.datasetName.group('redshift')
                if z is None:
                    errMsg = funcname+"(): no HDF5 output specified. Either specify the redshift "+\
                             "of the output or include the redshift in the dataset name."
                    raise RunTimeError(errMsg)
                self.setHDF5Output(z)        
        # Compute optical depths
        self.setOpticalDepths(**kwargs)
        # Extract dust free luminosities
        luminosityName = datasetName.replace(self.datasetName.group('dust'),"")
        luminosity = np.copy(np.array(self.hdf5Output["nodeData/"+luminosityName]))
        recentLuminosity = None
        if "dustAtlasClouds" in self.datasetName.group('dust'):
            if "LineLuminosity" in self.datasetName.group(0):
                recentName = luminosityName
            else:
                recentName = datasetName.replace(self.datasetName.group('dust'),"recent")
            recentLuminosity = np.copy(np.array(self.hdf5Output["nodeData/"+recentName]))
        self.attenuatedLuminosity = self.applyAttenuation(luminosity,recentLuminosity=recentLuminosity)
        return
                
    def getAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName.startswith("total"):
            luminosity = self.getAttenuatedLuminosity(datasetName.replace("total","disk"),overwrite=overwrite,z=z,**kwargs) + \
                self.getAttenuatedLuminosity(datasetName.replace("total","spheroid"),overwrite=overwrite,z=z,**kwargs)
        else:
            self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,**kwargs)
            luminosity = self.attenuatedLuminosity
        return luminosity


    def writeLuminosityToFile(self,overwrite=False):
        if not self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) or overwrite:
            out = self.galHDF5Obj.selectOutput(self.redshift)
            # Add luminosity to file
            self.galHDF5Obj.addDataset(out.name+"/nodeData/",self.datasetName.group(0),np.copy(self.attenuatedLuminosity))
            # Add appropriate attributes to new dataset
            if fnmatch.fnmatch(self.datasetName.group(0),"*LineLuminosity*"):
                attr = {"unitsInSI":luminositySolar}
            else:
                attr = {"unitsInSI":luminosityAB}
            self.galHDF5Obj.addAttributes(out.name+"/nodeData/"+self.datasetName.group(0),attr)
        return


