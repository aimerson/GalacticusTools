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

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        searchString = "^(?<component>disk|spheroid|total)(?<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
            ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?(?P<dust>:dust[^:]+)?$"
        MATCH = re.search(searchString,datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return MATCH

    def getLuminosityConversionFactor(self,continuumName):
        conversion = (luminosityAB/plancksConstant/self.continuumUnits)
        conversion *= np.log(self.wavelengthRanges[continuumName][1]/self.wavelengthRanges[continuumName][0])
        return conversion

    def getIonizingLuminosity(self,galHDF5Obj,z,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        MATCH = self.setDatasetName(datasetName)
        component = MATCH.group('component')
        continuumName = MATCH.group('continuum')
        redshift = MATCH.group('redshift')
        recent = MATCH.group('recent')
        if not recent:
            recent = ""
        dust = MATCH.group('dust')
        if not recent:
            dust = ""
        # Get appropriate stellar luminosity
        OUT = galHDF5Obj.selectOutput(z)
        luminosityName = component+"LuminositiesStellar:"+self.filterNames[continuumName]+":rest:z"+str(redshift)+recent+dust   
        if component == "total":
            luminosity = self.computeIonizingLuminosity(galHDF5Obj,z,luminosityName.replace("total","disk")) + \
                self.computeIonizingLuminosity(galHDF5Obj,z,luminosityName.replace("total","spheroid"))
        else:
            if not galHDF5Obj.datasetExists(luminosityName,z):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            luminosity = np.copy(OUT["nodeData/"+luminosityName])*self.getLuminosityConversion(continuumName)
        return luminosity
    
