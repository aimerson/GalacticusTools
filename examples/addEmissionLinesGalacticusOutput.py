#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from galacticus.utils.progress import Progress
from galacticus.io import GalacticusHDF5
from galacticus.derivedProperties import derivedProperties
from galacticus.EmissionLines import getLineNames



hdf5File = sys.argv[1]
emissionLines = getLineNames()

# Open HDF5 file for updating
GALACTICUS = GalacticusHDF5(hdf5File,'a')
DERIV = derivedProperties(GALACTICUS,verbose=True)

# Loop over outputs
redshifts = GALACTICUS.outputs.z
for i,z in enumerate(redshifts):
    print "-----> Processing z = "+str(z)+"  (iz = "+str(i+1)+"/"+str(len(redshifts))+")"
    # Get list of available properties at this redshift
    allDatasets = GALACTICUS.availableDatasets(z)
    if len(allDatasets) == 0:
        continue
    # Select this output
    OUT = GALACTICUS.selectOutput(z)
    # Get redshift string
    zStr = GALACTICUS.getRedshiftString(z)
    # Get list of emission line luminosity options
    lines = ["totalLineLuminosity:"+line+":rest:"+zStr for line in emissionLines] +\
        ["totalLineLuminosity:"+line+":observed:"+zStr for line in emissionLines]
    # Compute derived properties
    DERIV.addDatasets(z,lines,overwrite=False)
    
GALACTICUS.fileObj.close()


