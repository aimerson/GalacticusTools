#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError

def Get_StellarMass(galHDF5Obj,z,overwrite=False):    
    """
    Get_StellarMass(): Calculate and store stellar mass for whole galaxy. This
                       has the property name 'massStellar'.

    USAGE: mass = Get_StellarMass(galHDF5Obj,z,[overwrite])

           Inputs:
   
                 galHDF5Obj : Instance of GalacticusHDF5 file object.
                 z          : Redshift of output to work with.
                 overwrite  : Overwrite any existing value for total stellar
                              mass. (Default value = False).

            Outputs:
                 
                 mass       : Numpy array of total stellar masses.                    

    """
    funcname = sys._getframe().f_code.co_name
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if total stellar mass already calculated
    if "massStellar" in galHDF5Obj.availableDatasets(z) and not overwrite:        
        return np.array(out["nodeData/massStellar"])    
    # Extract stellar mass components and calculate total mass
    data = galHDF5Obj.readGalaxies(z,props=["spheroidMassStellar","diskMassStellar"])        
    totalStellarMass = data["diskMassStellar"] + data["spheroidMassStellar"]
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","massStellar",totalStellarMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskMassStellar"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/massStellar",attr)
    return totalStellarMass



def Get_StarFormationRate(galHDF5Obj,z,overwrite=False):    
    """
    Get_StarFormationRate(): Calculate and store star formation rate for whole galaxy. 
                             This has the property anme 'starFormationRate'.

    USAGE: mass = Get_StarFormationRate(galHDF5Obj,z,[overwrite])

           Inputs:
   
                 galHDF5Obj : Instance of GalacticusHDF5 file object.
                 z          : Redshift of output to work with.
                 overwrite  : Overwrite any existing value for total star formation 
                              rate. (Default value = False).

            Outputs:
                 
                 sfr        : Numpy array of total star formation rates.                    

    """
    funcname = sys._getframe().f_code.co_name
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if total SFR already calculated
    if "starFormationRate" in galHDF5Obj.availableDatasets(z) and not overwrite:        
        return np.array(out["nodeData/starFormationRate"])    
    # Extract SFR components and calculate total SFR
    data = galHDF5Obj.readGalaxies(z,props=["spheroidStarFormationRate","diskStarFormationRate"])        
    totalStellarMass = data["diskStarFormationRate"] + data["spheroidStarFormationRate"]
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","starFormationRate",totalStellarMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskStarFormationRate"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/starFormationRate",attr)
    return totalStellarMass
    

