#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .constants import luminosityAB
from .constants import luminositySolar,erg
from .GalacticusErrors import ParseError
from .uitls.progress import Progress

def ergPerSecond(luminosity):    
    luminosity = np.log10(luminosity)
    luminosity += np.log10(luminositySolar)
    luminosity -= np.log10(erg)
    luminosity = 10.0**luminosity
    return luminosity


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
