#! /usr/bin/env python

import sys,os
import numpy as np
import xml.etree.ElementTree as ET

from ..io import GalacticusHDF5
from ..GalacticusErrors import ParseError
from ..config import *
from ..utils.progress import Progress


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



class dustAtlas(object):
    
    def __init__(self,verbose=False,debug=False):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose        
        self.debug = debug
        # Load dust atlas file
        self.dustFile = galacticusPath + "data/dust/atlasFerrara2000/attenuations_MilkyWay_dustHeightRatio1.0.xml"
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
            else:
                if self.verbose:
                    print(classname+"(): Loading disk attenuations...")
                self.diskAttenuation = loadDiskAttenuation(comp,verbose=self.verbose)
        # Initialise classes for emission lines and filters
        self.emissionLinesClass = emissionLines()
        self.filtersDatabase = GalacticusFilters()
        return

    
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
        if self.debug:
            print(funcname+"(): Processing dataset '"+datasetName+"'")
        if not fnmatch.fnmatch(datasetName,"*:dustAtlas*"):
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Check if computing face on attenuation
        dustExtension = ":dustAltas"
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
        if self.debug:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift
        # Compute effective wavelength for filter/line
        if emissionLineFlag:
            # i) emission lines
            effectiveWavelength = self.emissionLinesClass.getWavelength(filter)
            if debug:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Emission line filter information:\n        "+infoLine)
        else:
            # ii) photometric filters
            effectiveWavelength = self.filtersDatabase.getEffectiveWavelength(filter,verbose=self.debug)
            if frame == "observed":
                effectiveWavelength /= (1.0+float(redshift))
            if self.debug:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Photometric filter information:\n        "+infoLine)
        
