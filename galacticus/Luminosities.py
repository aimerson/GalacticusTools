#! /usr/bin/env python

import sys
import numpy as np
from .io import GalacticusHDF5
from .constants import luminosityAB
from .GalacticusErrors import ParseError

def Get_Luminosity(galHDF5Obj,z,datasetName,overwrite=False):
    funcname = sys._getframe().f_code.co_name
    # Check dataset name correspnds to a luminosity
    if not fnmatch.fnmatch(datasetName,"*LuminositiesStellar:*:z*"):
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Ensure never overwrite disk/spheroid luminosities
    if datasetName.startswith("disk") or datasetName.startswith("spheroid"):
        overwrite = False
    # Check if luminosity already calculated
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        return np.array(out["nodeData/"+datasetName])
    # If get to here must be a total luminosity -- compute total luminosity
    diskName = datasetName.replace("total","disk")
    spheroidName = datasetName.replace("total","spheroid")
    totalLuminosity = np.array(out["nodeData/"+diskName]) +\
        np.array(out["nodeData/"+spheroidName])
    # Write luminosity to file                                                                                                                                                            
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,totalLuminosity)
    attr = {"unitsInSI":luminosityAB}
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
    return totalLuminosity

