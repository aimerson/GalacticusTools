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
    funcname = sys._getframe().f_code.co_name
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


class IonizingContinuuaClass(object):

    def __init__(self,datasetName=None,luminosity=None,\
                     redshift=None,outputName=None):
        self.datasetName = datasetName
        self.luminosity = luminosity
        self.redshift = redshift
        self.outputName = outputName
        return

    def reset(self):
        self.datasetName = None
        self.luminosity = None
        self.redshift = None
        self.outputName = None
        return


def parseConinuuaLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract information from dataset name                                                                                                                                 
    searchString = "^(?P<component>disk|spheroid|total)(?P<continuum>Lyman|Helium|Oxygen)ContinuumLuminosity"+\
        ":z(?P<redshift>[\d\.]+)(?P<recent>:recent)?(?P<dust>:dust[^:]+)?$"    
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class IonizingContinuua(IonizingContinuuaBase):

    def __init__(self,galHDF5Obj,verbose=False):            
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(IonizingContinuua, self).__init__(verbose=verbose)
        # Initialise variables to store Galacticus HDF5 object
        self.galHDF5Obj = galHDF5Obj
        return

    def ionizingLuminosityAvailable(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get redshift of appropriate HDF5 output
        MATCH = parseConinuuaLuminosity(datasetName)
        z = float(MATCH.group('redshift'))
        # Get list of all available datasets
        allProps = self.galHDF5Obj.availableDatasets(z)        
        # Check that any galaxy properties exist
        if len(allProps) == 0:
            return False
        # Check if dataset already in list of available properties
        if datasetName in allProps:
            return True
        # Check if can compute ionizing continuum luminosity from appropriate
        # stellar luminosities
        allLums = fnmatch.filter(allProps,"*LuminositiesStellar:*")
        if len(allLums) == 0:
            return False
        stellarLuminosityName = self.stellarLuminosityName(datasetName)
        if stellarLuminosityName in allLums:
            return True
        return False
        

    def stellarLuminosityName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = parseConinuuaLuminosity(datasetName)
        luminosityName = MATCH.group('component')+"LuminositiesStellar:"+self.filterNames[MATCH.group('continuum')]+\
            ":rest:z"+MATCH.group('redshift')
        if MATCH.group('recent') is not None:
            luminosityName = luminosityName + MATCH.group('recent')
        if MATCH.group('dust') is not None:
            luminosityName = luminosityName + MATCH.group('dust')
        return luminosityName

    def setIonizingLuminosity(self,datasetName,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store ionising continuua information
        IONCON = IonizingContinuuaClass()
        IONCON.datasetName = parseConinuuaLuminosity(datasetName)
        # Identify HDF5 output
        IONCON.outputName = self.galHDF5Obj.nearestOutputName(float(IONCON.datasetName.group('redshift')))
        HDF5OUT = self.galHDF5Obj.selectOutput(float(IONCON.datasetName.group('redshift')))
        # Check if luminosity already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(IONCON.datasetName.group('redshift'))) and not overwrite:
            IONCON.luminosity = np.array(HDF5OUT["nodeData/"+datasetName])
            return IONCON
        # Set line redshift
        if "lightconeRedshift" in HDF5OUT.keys():
            IONCON.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(IONCON.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            IONCON.redshift = np.ones(ngals,dtype=float)*z
        # Extract luminosity        
        luminosityName = self.stellarLuminosityName(datasetName)
        if datasetName.startswith("total"):
            IONDISK = self.setIonizingLuminosity(datasetName.replace("total","disk"),overwrite=overwrite)
            IONSPHERE = self.setIonizingLuminosity(datasetName.replace("total","spheroid"),overwrite=overwrite)
            IONCON.luminosity = np.copy(IONDISK.luminosity) + np.copy(IONSPHERE.luminosity)
            del IONDISK, IONSPHERE
        else:
            if not self.galHDF5Obj.datasetExists(luminosityName,float(IONCON.datasetName.group('redshift'))):
                raise IndexError(funcname+"(): dataset '"+luminosityName+"' not found!")
            continuumName = IONCON.datasetName.group('continuum')
            IONCON.luminosity = np.copy(HDF5OUT["nodeData/"+luminosityName])*self.getLuminosityConversionFactor(continuumName)
        return IONCON

    def getIonizingLuminosity(self,datasetName,overwrite=False,z=None,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        IONCON = self.setIonizingLuminosity(datasetName,overwrite=overwrite)
        return IONCON.luminosity


