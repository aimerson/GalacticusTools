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



class dustHybrid(DustProperties):
    
    def __init__(self,verbose=False,interpolateBoundsError=False,interpolateFillValue=None,\
                     extrapolateInSize=True,extrapolateInTau=True,\
                     opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self._verbose = verbose        
        # Initialise DustProperties class
        super(dustHybrid, self).__init__()
        # Load dust atlas file
        self.dustFile = pkg_resources.resource_filename(__name__,"../data/dust/Ferrara2000/attenuations_MilkyWay_dustHeightRatio1.0.xml")
        if not os.path.exists(self.dustFile):
            raise IOError(classname+"(): Cannot find Ferrara et al. (2000) dust atlas file!")
        else:
            if self._verbose:
                print(classname+"(): Loading Ferrara et al. (2000) dust atlas file...")
        self.dustData = ET.parse(self.dustFile)
        # Construct map of XML parents/descendents
        self.dustRoot = self.dustData.getroot()
        self.dustMap = {c.tag:p for p in self.dustRoot.iter() for c in p}        
        # Load wavelengths
        if self._verbose:
            print(classname+"(): Extracting wavelengths...")
        self.wavelengths = np.copy([ float(lam.text) for lam in self.dustMap["wavelengths"].iter("lambda")])        
        # Load attenuations for components
        self.diskAttenuation = None
        self.spheroidAttenuation = None
        for comp in self.dustData.iter("components"):
            if comp.find("name").text == "bulge":                
                if self._verbose:
                    print(classname+"(): Loading spheroid attenuations...")
                self.spheroidAttenuation = loadSpheroidAttenuation(comp,verbose=self._verbose)                
                axes = (self.spheroidAttenuation["size"],self.spheroidAttenuation["inclination"],\
                            self.spheroidAttenuation["opticalDepth"],self.wavelengths)                
                self.spheroidInterpolator = RegularGridInterpolator(axes,self.spheroidAttenuation["attenuation"],\
                                                                        bounds_error=interpolateBoundsError,\
                                                                        fill_value=interpolateFillValue)
            else:
                if self._verbose:
                    print(classname+"(): Loading disk attenuations...")
                self.diskAttenuation = loadDiskAttenuation(comp,verbose=self._verbose)
                axes = (self.diskAttenuation["inclination"],self.diskAttenuation["opticalDepth"],self.wavelengths)
                self.diskInterpolator = RegularGridInterpolator(axes,self.diskAttenuation["attenuation"],\
                                                                    bounds_error=interpolateBoundsError,\
                                                                    fill_value=interpolateFillValue)                                                                
        # Set options whether to extrapolate for various properties
        self.extrapolateInSize = extrapolateInSize
        self.extrapolateInTau = extrapolateInTau
        # Initialise classes for emission lines and filters
        self.emissionLinesClass = GalacticusEmissionLines()
        self.filtersDatabase = GalacticusFilters()
        # Specify constants in extinction model
        # -- "wavelengthExponent" is set to the value of 0.7 found by Charlot & Fall (2000).
        # -- "opticalDepthCloudsFactor" is set to unity, such that in gas with Solar metallicity the cloud optical depth will be 1.
        # -- "opticalDepthISMFactor" is set to 1.0 such that we reproduce the standard (Bohlin et al 1978) relation between visual
        #     extinction and column density in the local ISM (essentially solar metallicity).
        self.opticalDepthISMFactor = 1.0
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        return


    def InterpolateDustTable(self,component,wavelength,inclination,opticalDepth,bulgeSize=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if component.lower() == "disk":
            attenuation = self.diskInterpolator(zip(inclination,opticalDepth,wavelength))
        elif component.lower() == "spheroid":
            if bulgeSize is None:
                raise TypeError(funcname+"(): For spheroid dust attenuation, need to provide bulge sizes!")
            attenuation = self.spheroidInterpolator(zip(bulgeSize,inclination,opticalDepth,wavelength))
        else:
            raise ValueError(funcname+"(): Value for 'component' not recognised! Must be either 'spheroid' or 'disk'!")
        
        #np.place(attenuation,attenuation>1.0,1.0)
        #np.place(attenuation,attenuation<0.0,0.0)
        return attenuation


    def getCloudGasMetalsSurfaceDensity(self,galHDF5Obj,z,component,mcloud=1.0e6,rcloud=16.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Get metal mass
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        gasMass = np.array(out["nodeData/"+component+"MassGas"])
        metallicity = gasMetalMass/gasMass
        # Compute surface density
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = metallicity*mcloud/(2.0*Pi*(mega*rcloud)**2)
        np.place(gasMetalsSurfaceDensityCentral,np.isnan(scaleLength),0.0)
        return gasMetalsSurfaceDensityCentral
    

    def getCentralGasMetalsSurfaceDensity(self,galHDF5Obj,z,component):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Get metal mass
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        # Get scale length
        scaleLength = np.array(out["nodeData/diskRadius"])
        np.place(scaleLength,scaleLength<=0.0,np.nan)
        # Compute surface density
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = gasMetalMass/(2.0*Pi*(mega*scaleLength)**2)
        np.place(gasMetalsSurfaceDensityCentral,np.isnan(scaleLength),0.0)
        return gasMetalsSurfaceDensityCentral

    
    def getBulgeSizes(self,galHDF5Obj,z,component):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Get bulge sizes
        if fnmatch.fnmatch(component,"spheroid"):
            spheroidMassDistribution = galHDF5Obj.parameters["spheroidMassDistribution"].lower()
            spheroidRadius = np.array(out["nodeData/spheroidRadius"])
            diskRadius = np.array(out["nodeData/diskRadius"])
            np.place(diskRadius,diskRadius<=0.0,1.0)
            if fnmatch.fnmatch(spheroidMassDistribution,"hernquist"):
                sizes = (1.0+np.sqrt(2.0))*spheroidRadius/diskRadius
            elif fnmatch.fnmatch(spheroidMassDistribution,"sersic"):
                sizes = spheroidRadius/diskRadius
            else:
                raise ValueError(funcname+"(): Value for parameter 'spheroidMassDistribution' must be either 'hernquist' or 'sersic'!")
            if not self.extrapolateInSize:
                sizeMinimum = self.spheroidAttenuation["size"].min()
                sizeMaximum = self.spheroidAttenuation["size"].max()
                np.place(sizes,sizes<sizeMinimum,sizeMinimum)
                np.place(sizes,sizes>sizeMaximum,sizeMaximum)
        else:
            sizes = None
        return sizes


    def computeAttenuation(self,galHDF5Obj,z,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Compute attenuation
        if self._verbose:
            print(funcname+"(): Processing dataset '"+datasetName+"'")            
        # Check is a luminosity for attenuation
        MATCH = re.search(r"^(disk|spheroid)(LuminositiesStellar|LineLuminosity):([^:]+):([^:]+):z([\d\.]+)(:contam_[^:]+)?:dustHybrid([^:]+)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract dataset information
        component = MATCH.group(1)
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        luminosityType = MATCH.group(2)        
        filter = MATCH.group(3)
        frame = MATCH.group(4)
        redshift = MATCH.group(5)
        if self._verbose:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)
        contamination = MATCH.group(6)
        if contamination is None:
            contamination = ""
        dustExtension = ":dustHybrid"
        dustOption = MATCH.group(7)
        faceOn = False
        includeClouds = True
        if dustOption is not None:
            dustExtension = dustExtension + dustOption
            if "noclouds" in dustOption.lower():
                includeClouds = False
            else:
                includeClouds = True
            if "faceon" in dustOption.lower():
                faceOn = True
            else:
                faceOn = False            
        # Get name of unattenuated dataset
        luminosityDataset = datasetName.replace(dustExtension,"")
        # Get name for luminosity from recent star formation
        if fnmatch.fnmatch(luminosityType,"LineLuminosity"):
            recentLuminosityDataset = luminosityDataset
        else:
            recentLuminosityDataset = luminosityDataset+":recent"
            if not recentLuminosityDataset in list(map(str,out["nodeData"].keys())):
                raise IOError(funcname+"(): Missing luminosity for recent star formation for filter '"+luminosityDataset+"'!")
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift+contamination
        # Compute effective wavelength for filter/line
        if frame == "observed":
            effectiveWavelength = self.effectiveWavelength(filter,redshift=float(redshift),verbose=self._verbose)
        else:
            effectiveWavelength = self.effectiveWavelength(filter,redshift=None,verbose=self._verbose)
        # Get inclinations
        if faceOn:
            inclinations = np.zeros_like(np.array(out["nodeData/diskRadius"]))
        else:
            inclinations = getInclination(galHDF5Obj,z,overwrite=False,returnDataset=True)
        # Get bulge sizes
        sizes = self.getBulgeSizes(galHDF5Obj,z,component)
        # Compute gas metallicity and central surface density in M_Solar/pc^2
        gasMass = np.array(out["nodeData/"+component+"MassGas"])
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        scaleLength = np.array(out["nodeData/"+component+"Radius"])
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        del gasMetalMass,scaleLength        
        # Compute central optical depths
        opticalDepthCentral = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)        
        del gasMetalsSurfaceDensityCentral
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
        # Interpolate dustAtlas table to get attenuations
        wavelengths = effectiveWavelength*np.ones_like(inclinations)
        attenuationISM = self.InterpolateDustTable(component,wavelengths,inclinations,opticalDepthCentral,bulgeSize=sizes)
        # Set attenuation to unity for galaxies with no disk
        diskRadius = np.array(out["nodeData/diskRadius"])
        np.place(attenuationISM,diskRadius<=0.0,1.0)       
        # Print warnings for any un-physical attenuations
        if any(attenuationISM>1.0) and self._verbose:
            print("WARNING! "+funcname+"(): Some attenuations greater than unity! This is not physical!")
        attenuationISM = np.minimum(attenuationISM,1.0)
        if any(attenuationISM<0.0) and self._verbose:
            print("WARNING! "+funcname+"(): Some attenuations less than zero! This is not physical!")            
        attenuationISM = np.maximum(attenuationISM,0.0)
        attenuationISM *= self.opticalDepthISMFactor
        # Compute Charlot & Fall attenuation for clouds
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepthClouds = self.opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        attenuationClouds = np.exp(-opticalDepthClouds)
        # Apply attenuations and return result 
        if fnmatch.fnmatch(luminosityType,"LuminositiesStellar"):
            # i) stellar luminosities
            result = np.array(out["nodeData/"+luminosityDataset]) - np.array(out["nodeData/"+recentLuminosityDataset])
            result += np.array(out["nodeData/"+recentLuminosityDataset])*attenuationClouds
            result *= attenuationISM
        else:
            # ii) emission lines
            result = np.array(out["nodeData/"+luminosityDataset])*attenuationClouds*attenuationISM
        return result


    def attenuate(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if dust attenuated luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            if progressObj is not None:
                progressObj.increment()
                progressObj.print_status_line()
            if returnDataset:
                out = galHDF5Obj.selectOutput(z)
                return np.array(out["nodeData/"+datasetName])
            else:
                return
        # Check if a total luminosity or disk/spheroid luminosity
        if datasetName.startswith("total"):
            diskResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","disk"),overwrite=False,returnDataset=True)
            spheroidResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","spheroid"),overwrite=False,returnDataset=True)
            result = np.copy(diskResult) + np.copy(spheroidResult)
            del diskResult,spheroidResult
        else:
            result = self.computeAttenuation(galHDF5Obj,z,datasetName)
        # Write property to file and return result
        out = galHDF5Obj.selectOutput(z)
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,result)
        attr = None
        if fnmatch.fnmatch(datasetName,"*LuminositiesStellar*"):
            attr = {"unitsInSI":luminosityAB}
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):
            attr = {"unitsInSI":luminositySolar}
        if attr is not None:
            galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return result
        return
