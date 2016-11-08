#! usr/bin/env python

import numpy as np
from .io import GalacticusHDF5
from .constants import Pi

def Generate_Random_Inclinations(N,degrees=True):
    """
    Generate_Random_Inclinations: Return a list of N random inclination angles.

    USAGE:     inc = Generate_Random_Inclinations(N,[degrees])

        Inputs

              N       : Integer number of angles to generate.
              degrees : Return angles in degrees? (Default value = True)
              
        Output

              inc     : Numpy array of N inclination angles.

    """
    angles = np.arccos(np.random.rand(N))
    if degrees:
        angles *= 180.0/Pi
    return 


def getInclination(galHDF5Obj,z,overwrite=False,returnDataset=True):
    """
    getInclination: Generate and store a list of inclinations for the 
                    galaxies at redshift, z.

    USAGE: inc = getInclination(galHDF5Obj,z,[overwrite],[returnDataset])
    
        Inputs

             galHDF5Obj    : Instance of GalacticusHDF5 file object.
             z             : Redshift of output to work with. 
             overwrite     : Overwrite any existing value for inclination.
                             (Default value = False)                                                                                                                                                                                            
             returnDataset : Return array of dataset values? (Default value = True)

        Outputs
   
            inc            : Numpy array of inclinations (if returnDataset=True).
    
    """
    out = galHDF5Obj.selectOutput(z)
    # Check if inclination already calculated
    if "inclination" in galHDF5Obj.availableDatasets(z) and not overwrite:
        return np.array(out["nodeData/inclination"])
    N = len(np.array(out["nodeData/nodeIndex"]))
    inclination = Generate_Random_Inclinations(N)
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/","inclination",inclination)
    if returnDataset:
        return inclination
    return

