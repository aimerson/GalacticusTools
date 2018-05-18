#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .Luminosities import LuminosityClass
from .constants import luminosityAB
from .GalacticusErrors import ParseError
from .utils.progress import Progress


class StellarLuminosityClass(LuminosityClass):
    
    def __init__(self,datasetName=None,luminosity=None,\
                     redshift=None,outputName=None):
        super(StellarLuminosityClass,self).__init__(datasetName=datasetName,luminosity=luminosity,\
                                                        redshift=redshift,outputName=outputName)        
        self.unitsInSI = luminosityAB
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.luminosity = None
        return

class BulgeToTotalClass(object):
    
    def __init__(self,datasetName=None,BTratio=None,\
                     redshift=None,outputName=None):
        self.datasetName = datasetName
        self.BTratio = BTratio
        self.redshift = redshift
        self.outputName = outputName
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.BTratio = None
        return


def parseStellarLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
    MATCH = re.search(searchString,datasetName)    
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH

def parseBulgeToTotal(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "bulgeToTotalLuminosities:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
    MATCH = re.search(searchString,datasetName)    
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")    
    return MATCH


class StellarLuminosities(object):
    
    def __init__(self,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.unitsInSI = luminosityAB
        return

    def availableLuminosities(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        properties = self.galHDF5Obj.availableDatasets(z)
        return fnmatch.filter(properties,"*LuminositiesStellar:*")

    def createStellarLuminosityClass(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create class to store dust optical depths information
        LUM = StellarLuminosityClass()
        LUM.datasetName = parseStellarLuminosity(datasetName)
        # Identify HDF5 output
        LUM.outputName = self.galHDF5Obj.nearestOutputName(float(LUM.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(LUM.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            LUM.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(LUM.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            LUM.redshift = np.ones(ngals,dtype=float)*z
        return LUM

    def setLuminosity(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create luminosity class
        LUM = self.createStellarLuminosityClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(LUM.datasetName.group('redshift')))
        # Check if luminosity already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(LUM.datasetName.group('redshift'))) and not overwrite:
            LUM.luminosity = np.array(HDF5OUT["nodeData/"+datasetName])
            return LUM
        # Check whether dust attenuation is required
        if LUM.datasetName.group('dust') is not None:
            raise RuntimeError(funcname+"(): cannot load dataset as dust attenuation has not yet been carried out.")
        # Compute luminosity
        if datasetName.startswith("total"):
            DISKLUM = self.setLuminosity(datasetName.replace("total","disk"),z=z,overwrite=overwrite)
            SPHERELUM = self.setLuminosity(datasetName.replace("total","spheroid"),z=z,overwrite=overwrite)
            LUM.luminosity = np.copy(DISKLUM.luminosity) + np.copy(SPHERELUM.luminosity)
            del DISKLUM,SPHERELUM
        else:            
            LUM.luminosity = np.copy(np.array(HDF5OUT["nodeData/"+LUM.datasetName.group(0)]))
        return LUM

    def getLuminosity(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name    
        LUM = self.setLuminosity(datasetName,z=z,overwrite=overwrite)                
        return LUM.luminosity
        

    def createBulgeToTotalClass(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create class to store dust optical depths information
        RATIO = BulgeToTotalClass()
        RATIO.datasetName = parseBulgeToTotal(datasetName)
        # Identify HDF5 output
        RATIO.outputName = self.galHDF5Obj.nearestOutputName(float(RATIO.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(RATIO.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            RATIO.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(RATIO.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            RATIO.redshift = np.ones(ngals,dtype=float)*z
        return RATIO

    def setBulgeToTotal(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create dust class
        RATIO = self.createBulgeToTotalClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(RATIO.datasetName.group('redshift')))
        # Check if dataset already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(RATIO.datasetName.group('redshift'))) and not overwrite:
            RATIO.BTratio = np.array(HDF5OUT["nodeData/"+datasetName])
            return RATIO
        # Build names of luminosities
        spheroidName = "spheroidLuminositiesStellar:"+RATIO.datasetName.group('filterName')+":"\
            +RATIO.datasetName.group('frame')+":"+RATIO.datasetName.group('redshiftString')
        if RATIO.datasetName.group('recent') is not None:
            spheroidName = spheroidName + RATIO.datasetName.group('recent')
        if RATIO.datasetName.group('dust') is not None:
            spheroidName = spheroidName + RATIO.datasetName.group('dust')
        # Extract luminosities
        bulgeLum = self.getLuminosity(spheroidName,z=z,overwrite=overwrite)
        totalLum = self.getLuminosity(spheroidName.replace("spheroid","total"),z=z,overwrite=overwrite)
        # Compute ratio
        RATIO.BTratio = np.copy(buldgeLum)/np.copy(totalLum)
        del bulgeLum,totalLum
        return RATIO

    def getBulgeToTotal(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        RATIO = self.setBulgeToTotal(datasetName,z=z,overwrite=overwrite)
        return RATIO.BTratio

