#! /usr/bin/env python

import sys,re
import fnmatch
import numpy as np
import pkg_resources
from .hdf5 import HDF5
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .Filters import getFilterTransmission
from .constants import luminosityAB,plancksConstant



def getWavelengthLimits(filterFile):
    transmission = getFilterTransmission(filterFile)
    mask = transmission.transmission > 0.0
    transmission = transmission[mask]
    minWavelength = np.maximum(0.0,transmission.wavelength.min())
    maxWavelength = np.minimum(1.0e30,transmission.wavelength.max())
    return (minWavelength,maxWavelength)
    

class IonizingContinuua(object):
    def __init__(self):            
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set continuum units in photons/s
        self.continuumUnits = 1.0000000000000000e+50 
        # Load wavelength ranges for different continuua
        self.wavelengthRanges = {}
        # Set filter names
        self.filterNames = {"Lyman":"Lyc","Helium":"HeliumContinuum","Oxygen":"OxygenContinuum"}
        # i) Lyman continuum
        filterFile = pkg_resources.resource_filename(__name__,"data/filters/"+self.filterNames["Lyman"]+".xml")
        self.wavelengthRanges["Lyman"] = getWavelengthLimits(filterFile)
        # ii) Helium
        filterFile = pkg_resources.resource_filename(__name__,"data/filters/"+self.filterNames["Helium"]+".xml")
        self.wavelengthRanges["Helium"] = getWavelengthLimits(filterFile)
        # ii) Oxygen
        filterFile = pkg_resources.resource_filename(__name__,"data/filters/"+self.filterNames["Oxygen"]+".xml")
        self.wavelengthRanges["Oxygen"] = getWavelengthLimits(filterFile)        
        return

    
    def computeIonizingLuminosity(self,galHDF5Obj,z,datasetName,postProcessingInformation=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        MATCH = re.search(r"^(disk|spheroid|total)(Lyman|Helium|Oxygen)ContinuumLuminosity:z([\d\.]+)",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        component = MATCH.group(1)
        continuumName = MATCH.group(2)
        redshift = MATCH.group(3)
        # Get appropriate stellar luminosity
        if postProcessingInformation is not None:
            if not postProcessingInformation.startswith(":"):
                postProcessingInformation = ":"+postProcessingInformation
        else:
            postProcessingInformation = ""
        OUT = galHDF5Obj.selectOutput(z)
        if component == "total":
            luminosityName = "diskLuminositiesStellar:"+self.filterNames[continuumName]+":rest:z"+str(redshift)+postProcessingInformation
            if not galHDF5Obj.datasetExists(luminosityName,z):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            luminosity = np.copy(OUT["nodeData/"+luminosityName])
            luminosityName = "spheroidLuminositiesStellar:"+self.filterNames[continuumName]+":rest:z"+str(redshift)+postProcessingInformation
            if not galHDF5Obj.datasetExists(luminosityName,z):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            luminosity += np.copy(OUT["nodeData/"+luminosityName])
        else:
            luminosityName = component+"LuminositiesStellar:"+self.filterNames[continuumName]+":rest:z"+str(redshift)+postProcessingInformation
            if not galHDF5Obj.datasetExists(luminosityName,z):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            luminosity = np.copy(OUT["nodeData/"+luminosityName])
        conversion = (luminosityAB/plancksConstant/self.continuumUnits)
        conversion *= np.log10(self.wavelengthRanges[continuumName][1]/self.wavelengthRanges[continuumName][0])
        return luminosity*conversion
        
