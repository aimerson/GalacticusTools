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
    

class IonizingContinuuaBase(object):

    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
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

    def getLuminosityConversionFactor(self,continuumName):
        conversion = (luminosityAB/plancksConstant/self.continuumUnits)
        conversion *= np.log(self.wavelengthRanges[continuumName][1]/self.wavelengthRanges[continuumName][0])
        return conversion

    def getStellarLuminosityName(self,datasetName):
        searchString = "^(?P<component>disk|spheroid|total)(?P<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
                       ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?(?P<dust>:dust[^:]+)?$"
        MATCH = re.search(searchString,datasetName)
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+self.filterNames[MATCH.group('continuum')]+\
                         ":rest:z"+MATCH.group('redshift')
        if MATCH.group('recent') is not None:
            luminosityName = luminosityName + MATCH.group('recent')
        if MATCH.group('dust') is not None:
            luminosityName = luminosityName + MATCH.group('dust')            
        return luminosityName

class IonizingContinuua(IonizingContinuuaBase):

    def __init__(self,galHDF5Obj,verbose=False):            
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(IonizingContinuua, self).__init__(verbose=verbose)
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        # Initialise variables to store line information
        self.datasetName = None
        self.luminosity = None
        return

    def resetHDF5Output(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.redshift = None
        self.hdf5Output = None
        return

    def setHDF5Output(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        return

    def resetLuminosityInformation(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.datasetName = None
        self.luminosity = None
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        searchString = "^(?P<component>disk|spheroid|total)(?P<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
                       ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?(?P<dust>:dust[^:]+)?$"
        self.datasetName = re.search(searchString,datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return 

    def setIonizingLuminosity(self,datasetName,overwrite=False,z=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Reset luminosity information
        self.resetLuminosityInformation()
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
        # Check if luminosity already calculated
        if self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) and not overwrite:
            self.luminosity = np.array(out["nodeData/"+datasetName])
            return
        # Extract luminosity        
        luminosityName = self.getStellarLuminosityName(self.datasetName.group(0))
        if self.datasetName.group('component') == "total":
            luminosity = self.getIonizingLuminosity(self.datasetName.group(0).replace("total","disk")) + \
                self.getIonizingLuminosity(self.datasetName.group(0).replace("total","spheroid"))
        else:
            if not self.galHDF5Obj.datasetExists(luminosityName,self.redshift):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            continuumName = self.datasetName.group('continuum')
            luminosity = np.copy(self.hdf5Output["nodeData/"+luminosityName])*self.getLuminosityConversionFactor(continuumName)
        self.luminosity = luminosity
        return

    def getIonizingLuminosity(self,datasetName,overwrite=False,z=None,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.setIonizingLuminosity(datasetName,overwrite=overwrite,z=z)
        return self.luminosity


