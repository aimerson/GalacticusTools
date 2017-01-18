#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from galacticus.utils.progress import Progress
from galacticus.io import GalacticusHDF5
from galacticus.dust.Ferrara2000 import dustAtlas
from galacticus.dust.CharlotFall2000 import CharlotFall2000
from galacticus.dust.screens import dustScreen

hdf5File = sys.argv[1]

ATLAS = dustAtlas()
CHARLOTFALL = CharlotFall2000()
SLAB = dustScreen()

# Open HDF5 file for updating
GALACTICUS = GalacticusHDF5(hdf5File,'a')

# Loop over outputs
redshifts = GALACTICUS.outputs.z
for i,z in enumerate(redshifts):

    print "-----> Processing z = "+str(z)+"  (iz = "+str(i+1)+"/"+str(len(redshifts))+")"

    # Get list of available properties at this redshift
    allDatasets = GALACTICUS.availableDatasets(z)
    if len(allDatasets) == 0:
        continue
    
    visibleDatasets = list(np.unique(fnmatch.filter(allDatasets,"*LineLuminosity*"))) + \
        list(np.unique(fnmatch.filter(allDatasets,"*LuminositiesStellar*")))

    if len(visibleDatasets) == 0:
        continue
    
    # Select this output
    OUT = GALACTICUS.selectOutput(z)

    # Charlot & Fall 2000
    recentDatasets = list(np.unique(fnmatch.filter(visibleDatasets,"*:recent*")))
    recentDatasets = list(set(recentDatasets).difference(fnmatch.filter(recentDatasets,"*:dust*")))
    recentDatasets = " ".join(recentDatasets).replace(":recent",":dustCharlotFall2000").split()
    dummy = [CHARLOTFALL.attenuate(GALACTICUS,z,name) for name in recentDatasets]
    del dummy
    
    # Dust Atlas
    atlasDatasets = list(set(visibleDatasets).difference(fnmatch.filter(visibleDatasets,"*:recent*")))
    atlasDatasets = list(set(atlasDatasets).difference(fnmatch.filter(atlasDatasets,"*:dust*")))
    dummy = [ATLAS.attenuate(GALACTICUS,z,name+":dustAtlas") for name in atlasDatasets]
    del dummy

    # Dust screens
    screenDatasets = list(set(visibleDatasets).difference(fnmatch.filter(visibleDatasets,"*:recent*")))
    screenDatasets = list(set(screenDatasets).difference(fnmatch.filter(screenDatasets,"*:dust*")))
    dummy = [SLAB.attenuate(GALACTICUS,z,name+":dustScreen_calzetti2000_age2.0_Av0.1") for name in screenDatasets]
    del dummy
    dummy = [SLAB.attenuate(GALACTICUS,z,name+":dustScreen_seaton1979_age2.0_Av0.1") for name in screenDatasets]
    del dummy


GALACTICUS.fileObj.close()



