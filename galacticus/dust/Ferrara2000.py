#! /usr/bin/env python

import sys,os
import numpy as np
import xml.etree.ElementTree as ET

from ..config import *









def loadDiskAttenuation(component,verbose=False):               
    data = {}
    data["attenuation"] = None       
    # Loop over inclinations
    inclinations = component.findall("inclination")            
    data["inclination"] = np.zeros(len(inclinations),dtype=float)    
    inclinationCount = data["inclination"].size
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
    return data




class dustAtlas(object):
    
    def __init__(self,verbose=False):
        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose
        
        # Load dust atlas file
        self.dustFile = galacticusPath + "data/dust/atlasFerrara2000/attenuations_MilkyWay_dustHeightRatio1.0.xml"
        if not os.path.exists(self.dustFile):
            raise IOError(classname+"(): cannot find Ferrara et al. (2000) dust atlas file!")
        else:
            if self.verbose:
                print(classname+"(): loading Ferrara et al. (2000) dust atlas file...")
        self.dustData = ET.parse(self.dustFile)
        # Construct map of XML parents/descendents
        self.dustRoot = self.dustData.getroot()
        self.dustMap = {c.tag:p for p in self.dustRoot.iter() for c in p}        
        # Load wavelengths
        if self.verbose:
            print(classname+"(): extracting wavelengths...")
        self.wavelengths = np.copy([ float(lam.text) for lam in self.dustMap["wavelengths"].iter("lambda")])
        
        # Load attenuations for components
        for comp in self.dustData.iter("components"):
            if comp.find("name").text == "bulge":                
                pass
                self.spheroidAttenuations = loadSpheroidAttenuation(comp,verbose=self.verbose)
            else:
                if self.verbose:
                    print(classname+"(): loading disk attenuations...")
                self.diskAttenuation = loadDiskAttenuation(comp,verbose=self.verbose)
                
