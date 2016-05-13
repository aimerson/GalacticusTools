#! usr/bin/env python

import numpy as np
from .io import GalacticusHDF5

def Generate_Inclinations(N):
    return 180.0*np.arccos(np.random.rand(N))/3.1415927

def Get_Inclination(galHDF5Obj,z,overwrite=False):
    out = galHDF5Obj.selectOutput(z)
    # Check if inclination already calculated
    if "inclination" in galHDF5Obj.availableDatasets(z) and not overwrite:
        return np.array(out["nodeData/inclination"])
    N = len(np.array(out["nodeData/nodeIndex"]))
    inclination = Generate_Inclination(N)
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","inclination",inclination)
    return inclination

