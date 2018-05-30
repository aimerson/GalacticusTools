#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .galaxyProperties import DatasetClass
from .utils.progress import Progress
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .constants import massSolar



class StellarMassClass(DatasetClass):

    def __init__(self,datasetName=None,redshift=None,outputName=None,mass=None):
        super(StellarMassClass,self).__init__(datasetName=datasetName,redshift=redshift,\
                                                  outputName=outputName)
        self.mass = mass
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.mass = None
        return


class StarFormationRateClass(DatasetClass):

    def __init__(self,datasetName=None,redshift=None,outputName=None,sfr=None):
        super(StarFormationRateClass,self).__init__(datasetName=datasetName,redshift=redshift,\
                                                        outputName=outputName)
        self.sfr = sfr
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.sfr = None
        return

def parseStellarMass(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "(?P<component>\w+)MassStellar"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH

def parseStarFormationRate(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "(?P<component>\w+)StarFormationRate"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class GalacticusStellarMass(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.unitsInSI = massSolar
        return

    def __call__(self,datasetName,z):
        return self.getStellarMass(datasetName,z)
    
    def createStellarMassClass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store stellar mass information
        STARS = StellarMassClass()
        STARS.datasetName = parseStellarMass(datasetName)
        # Identify HDF5 output
        STARS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            STARS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(redshift)
            STARS.redshift = np.ones(ngals,dtype=float)*redshift
        return STARS

    def setStellarMass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create stellar mass class
        STARS = self.createStellarMassClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(z) and not overwrite:
            STARS.mass = np.array(HDF5OUT["nodeData/"+datasetName])
            return STARS
        # Compute stellar mass
        if datasetName.startswith("total"):
            STARS.mass = self.getStellarMass(datasetName.replace("total","disk")) +\
                self.getStellarMass(datasetName.replace("total","spheroid"))
        else:
            STARS.mass = np.copy(np.array(HDF5OUT["nodeData/"+STARS.datasetName.group(0)]))
        return STARS
            
    def getStellarMass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        STARS = self.setStellarMass(datasetName,z)
        return STARS.mass


class GalacticusStarFormationRate(object):

    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.Gyr = 60.0*60.0*24.*365*1.0e9
        self.unitsInSI = massSolar/self.Gyr
        return

    def __call__(self,datasetName,z):
        return self.getStarFormationRate(datasetName,z)


    def createStarFormationRateClass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store stellar mass information
        STARS = StarFormationRateClass()
        STARS.datasetName = parseStellarMass(datasetName)
        # Identify HDF5 output
        STARS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            STARS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(redshift)
            STARS.redshift = np.ones(ngals,dtype=float)*redshift
        return STARS

    def setStarFormationRate(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create stellar mass class
        STARS = self.createStarFormationRateClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(z) and not overwrite:
            STARS.sfr = np.array(HDF5OUT["nodeData/"+datasetName])
            return STARS
        # Compute stellar mass
        if datasetName.startswith("total"):
            STARS.mass = self.getStarFormationRate(datasetName.replace("total","disk")) +\
                self.getStarFormationRate(datasetName.replace("total","spheroid"))
        else:
            STARS.sfr = np.copy(np.array(HDF5OUT["nodeData/"+STARS.datasetName.group(0)]))
        return STARS

    def getStarFormationRate(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        STARS = self.setStarFormationRate(datasetName,z)
        return STARS.sfr
