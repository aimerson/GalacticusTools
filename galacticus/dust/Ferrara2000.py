#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
import pkg_resources
from scipy.interpolate import RegularGridInterpolator

from .utils import DustProperties
from ..io import GalacticusHDF5
from ..EmissionLines import emissionLines
from ..GalacticusErrors import ParseError
from ..Filters import GalacticusFilters
from ..config import *
from ..utils.progress import Progress
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



class dustAtlas(DustProperties):
    
    def __init__(self,verbose=False,interpolateBoundsError=False,interpolateFillValue=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self._verbose = verbose        

        # Initialise DustProperties class
        super(dustAtlas,self).__init__()

        # Load dust atlas file
        self.dustFile = pkg_resources.resource_filename(__name__,"data/dust/atlasFerrara2000/attenuations_MilkyWay_dustHeightRatio1.0.xml")
        if not os.path.exists(self.dustFile):
            raise IOError(classname+"(): Cannot find Ferrara et al. (2000) dust atlas file!")
        else:
            if self.verbose:
                print(classname+"(): Loading Ferrara et al. (2000) dust atlas file...")
        self.dustData = ET.parse(self.dustFile)
        # Construct map of XML parents/descendents
        self.dustRoot = self.dustData.getroot()
        self.dustMap = {c.tag:p for p in self.dustRoot.iter() for c in p}        
        # Load wavelengths
        if self.verbose:
            print(classname+"(): Extracting wavelengths...")
        self.wavelengths = np.copy([ float(lam.text) for lam in self.dustMap["wavelengths"].iter("lambda")])        
        # Load attenuations for components
        self.diskAttenuation = None
        self.spheroidAttenuation = None
        for comp in self.dustData.iter("components"):
            if comp.find("name").text == "bulge":                
                if self.verbose:
                    print(classname+"(): Loading spheroid attenuations...")
                self.spheroidAttenuation = loadSpheroidAttenuation(comp,verbose=self.verbose)                
                axes = (self.spheroidAttenuation["size"],self.spheroidAttenuation["inclination"],\
                            self.spheroidAttenuation["opticalDepth"],self.wavelengths)                
                self.spheroidInterpolator = RegularGridInterpolator(axes,self.spheroidAttenuation["attenuation"],\
                                                                        bounds_error=interpolateBoundsError,\
                                                                        fill_value=interpolateFillValue)
            else:
                if self.verbose:
                    print(classname+"(): Loading disk attenuations...")
                self.diskAttenuation = loadDiskAttenuation(comp,verbose=self.verbose)
                axes = (self.diskAttenuation["inclination"],self.diskAttenuation["opticalDepth"],self.wavelengths)
                self.diskInterpolator = RegularGridInterpolator(axes,self.diskAttenuation["attenuation"],\
                                                                    bounds_error=interpolateBoundsError,\
                                                                    fill_value=interpolateFillValue)                                                                
        # Initialise classes for emission lines and filters
        self.emissionLinesClass = emissionLines()
        self.filtersDatabase = GalacticusFilters()
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



    
    def attenuate(self,galHDF5Obj,z,datasetName,overwrite=False,\
                      extrapolateInSize=True,extrapolateInTau=True):
        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if dust attenuated luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])
        
        ####################################################################
        # Compute attenuation
        if self._verbose:
            print(funcname+"(): Processing dataset '"+datasetName+"'")
        if not fnmatch.fnmatch(datasetName,"*:dustAtlas*"):
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Set dust extension in dataset name        
        dustExtension = ":dustAtlas"        
        if ":dustatlasnoclouds" in datasetName.lower():
            dustExtension = dustExtension + "noClouds"        
        # Check if computing face on attenuation        
        faceOn = False
        if "faceOn" in datasetName:
            dustExtension = dustExtension+"faceOn"
            faceOn = True
        # Get name of unattenuated dataset
        luminosityDataset = datasetName.replace(dustExtension,"")
        # Get component of dataset (disk or spheroid)
        if luminosityDataset.startswith("disk"):
            component = "disk"
        elif luminosityDataset.startswith("spheroid"):
            component = "spheroid"
        else:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        # Extract dataset information
        datasetInfo = luminosityDataset.split(":")
        opticalDepthOrLuminosity = datasetInfo[0].replace(component,"")
        if "LineLuminosity" in datasetInfo[0]:
            emissionLineFlag = True
        else:
            emissionLineFlag = False
        filter = datasetInfo[1]
        frame = datasetInfo[2]
        redshift = datasetInfo[3].replace("z","")
        if self._verbose:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift
        # Compute effective wavelength for filter/line
        if emissionLineFlag:
            # i) emission lines
            effectiveWavelength = self.emissionLinesClass.getWavelength(filter)
            if self._verbose:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Emission line filter information:\n        "+infoLine)
        else:
            # ii) photometric filters
            effectiveWavelength = self.filtersDatabase.getEffectiveWavelength(filter,verbose=self._verbose)
            if frame == "observed":
                effectiveWavelength /= (1.0+float(redshift))
            if self._verbose:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Photometric filter information:\n        "+infoLine)
        # Get inclinations
        if faceOn:
            inclinations = np.zeros_like(np.array(out["nodeData/diskRadius"]))
        else:
            inclinations = np.array(out["nodeData/diskRadius"])
        # Get bulge sizes
        if fnmatch.fnmatch(component,"spheroid"):
            spheroidMassDistribution = galHDF5Obj.parameters["spheroidMassDistribution"].lower()
            if fnmatch.fnmatch(spheroidMassDistribution,"hernquist"):
                sizes = (1.0+np.sqrt(2.0))*np.array(out["nodeData/spheroidRadius"])/np.array(out["nodeData/diskRadius"])
            elif fnmatch.fnmatch(spheroidMassDistribution,"sersic"):
                sizes = np.array(out["nodeData/spheroidRadius"])/np.array(out["nodeData/diskRadius"])
            else:
                raise ValueError(funcname+"(): Value for parameter 'spheroidMassDistribution' must be either 'hernquist' or 'sersic'!")
            np.place(sizes,np.isnan(sizes),1.0)
            if not extrapolateInSize:
                sizeMinimum = self.spheroidAttenuation["size"].min()
                sizeMaximum = self.spheroidAttenuation["size"].max()
                np.place(sizes,sizes<sizeMinimum,sizeMinimum)
                np.place(sizes,sizes>sizeMaximum,sizeMaximum)
        else:
            sizes = None
        # Specify required constants for computing optical depth normalisation
        localISMMetallicity = 0.02  # ... Metallicity in the local ISM.
        AV_to_EBV = 3.10            # ... (A_V/E(B-V); Savage & Mathis 1979)
        NH_to_EBV = 5.8e21          # ... (N_H/E(B-V); atoms/cm^2/mag; Savage & Mathis 1979)
        opticalDepthToMagnitudes = 2.5*np.log10(np.exp(1.0)) # Conversion factor from optical depth to magnitudes of extinction.
        hecto = 1.0e2
        opticalDepthNormalization = (1.0/opticalDepthToMagnitudes)*(AV_to_EBV/NH_to_EBV)
        opticalDepthNormalization *= (massFractionHydrogen/massAtomic)*(massSolar/(Parsec*hecto)**2)/localISMMetallicity
        # Compute central surface density in M_Solar/pc^2
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        scaleLength = np.array(out["nodeData/diskRadius"])
        np.place(scaleLength,scaleLength<=0.0,np.nan)
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = gasMetalMass/(2.0*Pi*(mega*scaleLength)**2)
        np.place(gasMetalsSurfaceDensityCentral,np.isnan(scaleLength),0.0)
        # Compute central optical depths
        opticalDepthCentral = opticalDepthNormalization*gasMetalsSurfaceDensityCentral
        if not extrapolateInTau:
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
        attenuations = self.InterpolateDustTable(component,wavelengths,inclinations,opticalDepthCentral,bulgeSize=sizes)
        np.place(attenuations,np.isnan(scaleLength),1.0)
        if any(attenuations>1.0):
            print("WARNING! "+funcname+"(): Some attenuations greater than unity! This is not physical!")
        if any(attenuations<0.0):
            print("WARNING! "+funcname+"(): Some attenuations less than zero! This is not physical!")            
        # Apply attenuations and return/store result        
        result = np.array(out["nodeData/"+luminosityDataset])*attenuations
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,result)
        return result
        
