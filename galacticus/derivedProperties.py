#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .utils.progress import Progress

def addDerivedProperties(galHDF5Obj,z,datasets,overwrite=False,verbose=True):
    funcname = sys._getframe().f_code.co_name

    PROG = None
    if verbose:
        print(funcname+"(): processing derived datasets:")       
    # Stellar masses
    datasets = fnmatch.filter(datasets,"*MassStellar")
    if len(datasets)>0: 
        from .Stars import getStellarMass
        if verbose:
            print("    stellar mass...")
            PROG = Progress(len(datasets))
        dummy = [getStellarMass(galHDF5Obj,z,name,overwrite=overwrite,\
                                    returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy    
    # Star formation rates
    datasets = fnmatch.filter(datasets,"*StarFormationRate")
    if len(datasets)>0: 
        from .Stars import getStarFormationRate
        if verbose:
            print("    star formation rate...")
            PROG = Progress(len(datasets))
        dummy = [getStarFormationRate(galHDF5Obj,z,name,overwrite=overwrite,\
                                          returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy
    # Stellar luminosities 
    datasets = fnmatch.filter(datasets,"*LuminositiesStellar:*")
    datasets = list(set(datasets).difference(fnmatch.filter(datasets,"*:dust*")))
    if len(datasets)>0: 
        from .Stars import getLuminosity
        if verbose:
            print("    stellar luminosity...")
            PROG = Progress(len(datasets))
        dummy = [getLuminosity(galHDF5Obj,z,name,overwrite=overwrite,\
                                   returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy



