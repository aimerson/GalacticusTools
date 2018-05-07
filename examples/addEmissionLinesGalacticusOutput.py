#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from galacticus.postProcess import processGalacticusHDF5

# Get name of Galacticus file
hdf5File = sys.argv[1]

# Initialise class to post-process Galacticus file
GH5PP = processGalacticusHDF5(hdf5File,verbose=True)

# Get names of emission lines that can be calculated
emissionLines = GH5PP.emissionLines.getLineNames()
print("Available emission lines:")
for line in emissionLines:
    print("     -- "+line)
    
# Get redshifts in file
redshifts = GH5PP.outputs.z
print("Reshift outputs in file:")
for z in redshifts:
    print("    -- "+str(z))

# Loop over outputs
for i,z in enumerate(redshifts):
    ngals = GH5PP.countGalaxiesAtRedshift(z)
    # Check output contains galaxies
    if ngals > 0:
        print("Processing z = "+str(z)+"  (iz = "+str(i+1)+"/"+str(len(redshifts))+")")
        # Get redshift string for this output. The redshift string appears in various galaxy 
        # dataset names, e.g. totalLineLuminosity:balmerAlpha6563:rest:z1.500.
        zStr = GH5PP.getRedshiftString(z)
        # Construct names of emission line datasets to write to file
        emLines = ["totalLineLuminosity:"+name+":rest:"+zStr for name in lineNames] + \
                  ["totalLineLuminosity:"+name+":observed:"+zStr for name in lineNames]
        # Pass dataset for processing (calculation and writing to file)
        dummy = [GH5PP.processDataset(dset) for dset in emLines]

# Close file    
GH5PP.fileObj.close()


