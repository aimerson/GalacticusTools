#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import xml.etree.ElementTree as ET
import pkg_resources
from scipy.interpolate import RegularGridInterpolator
from ..config import *
from ..utils.progress import Progress


class dustAtlasTable(object):
    
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        self.dustFile = pkg_resources.resource_filename(__name__,"../data/dust/Ferrara1999/attenuations_MilkyWay_dustHeightRatio1.0.xml")
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

