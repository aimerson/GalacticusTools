#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .utils.progress import Progress
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .constants import massSolar


class GalacticusStellarMass(object):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.unitsInSI = massSolar
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = re.search("(?P<component>\w+)MassStellar",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return MATCH
        
    def getStellarMass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        hdf5Output = self.galHDF5Obj.selectOutput(z)
        if datasetName in self.galHDF5Obj.availableDatasets(z):
            return np.array(hdf5Output["nodeData/"+datasetName])
        datasetName = self.setDatasetName(datasetName)
        if fnmatch.fnmatch(datasetName.group('component'),"total"):
            stellarMass = self.getStellarMass("diskMassStellar",z) + \
                          self.getStellarMass("spheroidMassStellar",z)
        else:
            stellarMass = np.array(hdf5Output["nodeData/"+datasetName.group(0)])
        return stellarMass


def getStellarMass(galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):    
    """
    getStellarMass(): Calculate and store stellar mass for whole galaxy. This
                       has the property name 'totalMassStellar'.

    USAGE: mass = getStellarMass(galHDF5Obj,z,datasetName,[overwrite],[returnDataset])

           Inputs:
   
                 galHDF5Obj    : Instance of GalacticusHDF5 file object.
                 z             : Redshift of output to work with.
                 datasetName   : Name of dataset to return/process, i.e. (disk|spheroid|total)MassStellar.
                 overwrite     : Overwrite any existing value for total stellar
                                 mass. (Default value = False)
                 returnDataset : Return array of dataset values? (Default value = True)
                 progressObj   : Progress object instance to display progress bar if call is inside loop.
                                 If None, then progress not displayed. (Default value = None)

            Outputs:
                 
                 mass          : Numpy array of stellar masses (if returnDataset=True).                    

    """
    funcname = sys._getframe().f_code.co_name
    # Check dataset name is a stellar mass
    MATCH = re.search("(\w+)MassStellar",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if computing total stellar mass
    component = MATCH.group(1).lower()
    if component != "total":
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+component+"MassStellar"])    
    # Check if total stellar mass already calculated
    if "totalMassStellar" in galHDF5Obj.availableDatasets(z) and not overwrite:        
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/totalMassStellar"])    
    # Extract stellar mass components and calculate total mass
    data = galHDF5Obj.readGalaxies(z,props=["spheroidMassStellar","diskMassStellar"])        
    totalStellarMass = data["diskMassStellar"] + data["spheroidMassStellar"]
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","totalMassStellar",totalStellarMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskMassStellar"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/totalMassStellar",attr)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return totalStellarMass
    return



class GalacticusStarFormationRate(object):

    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.Gyr = 60.0*60.0*24.*365*1.0e9
        self.unitsInSI = massSolar/self.Gyr
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = re.search("(?P<component>\w+)StarFormationRate",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return MATCH

    def getStarFormationRate(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        hdf5Output = self.galHDF5Obj.selectOutput(z)
        if datasetName in self.galHDF5Obj.availableDatasets(z):
            return np.array(hdf5Output["nodeData/"+datasetName])
        datasetName = self.setDatasetName(datasetName)
        if fnmatch.fnmatch(datasetName.group('component'),"total"):
            sfr = self.getStellarMass("diskStarFormationRate",z) + \
                  self.getStellarMass("spheroidStarFormationRate",z)
        else:
            sfr = np.array(hdf5Output["nodeData/"+datasetName.group(0)])
        return sfr


def getStarFormationRate(galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):    
    """
    getStarFormationRate(): Calculate and store star formation rate for whole galaxy. 
                             This has the property name 'totalStarFormationRate'.

    USAGE: sfr = getStarFormationRate(galHDF5Obj,z,[overwrite],[returnDataset])

           Inputs:
   
                 galHDF5Obj    : Instance of GalacticusHDF5 file object.
                 z             : Redshift of output to work with.
                 datasetName   : Name of dataset to return/process, 
                                 i.e. (disk|spheroid|total)StarFormationRate.
                 overwrite     : Overwrite any existing value for total star formation 
                                 rate. (Default value = False).
                 returnDataset : Return array of dataset values? (Default value = True)
                 progressObj   : Progress object instance to display progress bar if call is inside loop.
                                 If None, then progress not displayed. (Default value = None)

            Outputs:
                 
                 sfr           : Numpy array of star formation rates (if returnDataset=True).                    

    """
    funcname = sys._getframe().f_code.co_name
    # Check dataset name is a star formation rate
    MATCH = re.search("(\w+)StarFormationRate",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Check if computing total SFR
    component = MATCH.group(1).lower()
    if component != "total":
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+component+"StarFormationRate"])    
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if total SFR already calculated
    if "totalStarFormationRate" in galHDF5Obj.availableDatasets(z) and not overwrite:        
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/totalStarFormationRate"])    
    # Extract SFR components and calculate total SFR
    data = galHDF5Obj.readGalaxies(z,props=["spheroidStarFormationRate","diskStarFormationRate"])        
    totalStellarMass = data["diskStarFormationRate"] + data["spheroidStarFormationRate"]
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","totalStarFormationRate",totalStellarMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskStarFormationRate"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/totalStarFormationRate",attr)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return totalStellarMass
    return
    

