#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .constants import luminosityAB
from .constants import luminositySolar,erg
from .GalacticusErrors import ParseError
from .utils.progress import Progress

def ergPerSecond(luminosity):    
    luminosity = np.log10(luminosity)
    luminosity += np.log10(luminositySolar)
    luminosity -= np.log10(erg)
    luminosity = 10.0**luminosity
    return luminosity


class StellarLuminosities(object):
    
    def __init__(self,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.unitsInSI = luminosityAB
        return

    def getLuminosity(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        OUT = self.galHDF5Obj.selectOutput(z)
        if self.galHDF5Obj.datasetExists(datasetName,z) and not overwrite:
            return np.array(OUT["nodeData/"+datasetName])
        # Check dataset name corresponds to a luminosity
        searchString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
                       "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
        MATCH = re.search(searchString,datasetName)    
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract luminosity
        if MATCH.group('component') == "total":
            luminosity = self.getLuminosity(datasetName.replace("total","disk"),z,overwrite=overwrite) + \
                         self.getLuminosity(datasetName.replace("total","spheroid"),z,overwrite=overwrite)
        else:
            luminosity = np.array(OUT["nodeData/"+datasetName])
        return luminosity
        
    def getBulgeToTotal(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        OUT = self.galHDF5Obj.selectOutput(z)
        if self.galHDF5Obj.datasetExists(datasetName,z) and not overwrite:
            return np.array(OUT["/nodeData/"+datasetName])
        # Check dataset name corresponds to a bulge-to-total ratio
        searchString = "bulgeToTotalLuminosities:(?P<filter>[^:]+):(?P<frame>[^:]+)"+\
                       "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<recent>:recent)?(?P<dust>:dust[^:]+)?"
        MATCH = re.search(searchString,datasetName)    
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        recent = MATCH.group('recent')
        if not recent:        
            recent = ""
        dust = MATCH.group('dust')
        if not dust:        
            dust = ""
        # Compute bulge-to-total luminosity
        spheroidName = "spheroidLuminositiesStellar:"+MATCH.group('filter')+":"+MATCH.group('frame')+":"+\
                       MATCH.group('redshiftString')+recent+dust
        totalName = "totalLuminositiesStellar:"+MATCH.group('filter')+":"+MATCH.group('frame')+":"+\
                       MATCH.group('redshiftString')+recent+dust
        ratio = self.getLuminosity(spheroidName,z,overwrite=overwrite)/self.getLuminosity(totalName,z,overwrite=overwrite)
        return ratio


def getLuminosity(galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
    funcname = sys._getframe().f_code.co_name
    # Check dataset name correspnds to a luminosity
    MATCH = re.search(r"^(disk|spheroid|total)LuminositiesStellar:([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?",datasetName)    
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Ensure never overwrite disk/spheroid luminosities
    component = MATCH.group(1)
    if component.lower() != "total":
        overwrite = False
    # Check if luminosity already calculated
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return np.array(out["nodeData/"+datasetName])
        else:
            return
    # If get to here must be a total luminosity -- compute total luminosity
    diskName = datasetName.replace("total","disk")
    spheroidName = datasetName.replace("total","spheroid")
    totalLuminosity = np.array(out["nodeData/"+diskName]) +\
        np.array(out["nodeData/"+spheroidName])
    # Write luminosity to file                                                                                                                                                            
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,totalLuminosity)
    attr = {"unitsInSI":luminosityAB}
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return totalLuminosity
    return


def getBulgeToTotal(galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
    funcname = sys._getframe().f_code.co_name
    # Check dataset name correspnds to a bulge-to-total luminosity
    MATCH = re.search(r"^bulgeToTotalLuminosities:([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if bulge-to-total luminosity already calculated
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return np.array(out["nodeData/"+datasetName])
        else:
            return
    # Compute bulge-to-total luminosity 
    diskName = datasetName.replace("bulgeToTotalLuminosities","diskLuminositiesStellar")
    spheroidName = datasetName.replace("bulgeToTotalLuminosities","spheroidLuminositiesStellar")
    ratio = np.array(out["nodeData/"+spheroidName])/ \
        (np.array(out["nodeData/"+diskName])+np.array(out["nodeData/"+spheroidName]))
    # Write luminosity to file
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,ratio)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return ratio
    return
