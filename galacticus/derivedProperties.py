#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .utils.progress import Progress

def addDerivedProperties(galHDF5Obj,z,derivedDatasets,overwrite=False,verbose=True):
    funcname = sys._getframe().f_code.co_name


    PROG = None
    if verbose:
        print(funcname+"(): processing derived datasets:")       
    # Stellar masses
    datasets = fnmatch.filter(derivedDatasets,"*MassStellar")
    if len(datasets)>0: 
        from .Stars import getStellarMass
        if verbose:
            print("    stellar mass...")
            PROG = Progress(len(datasets))
        dummy = [getStellarMass(galHDF5Obj,z,name,overwrite=overwrite,\
                                    returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy    
    # Star formation rates
    datasets = fnmatch.filter(derivedDatasets,"*StarFormationRate")
    if len(datasets)>0: 
        from .Stars import getStarFormationRate
        if verbose:
            print("    star formation rate...")
            PROG = Progress(len(datasets))
        dummy = [getStarFormationRate(galHDF5Obj,z,name,overwrite=overwrite,\
                                          returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy
    # Stellar luminosities 
    datasets = fnmatch.filter(derivedDatasets,"*LuminositiesStellar:*")
    datasets = list(set(datasets).difference(fnmatch.filter(datasets,"*:dust*")))
    if len(datasets)>0: 
        from .Stars import getLuminosity
        if verbose:
            print("    stellar luminosity...")
            PROG = Progress(len(datasets))
        dummy = [getLuminosity(galHDF5Obj,z,name,overwrite=overwrite,\
                                   returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy
    # Cold gas
    datasets = fnmatch.filter(derivedDatasets,"*MassGas") + fnmatch.filter(derivedDatasets,"*MassColdGas")
    if len(datasets)>0: 
        from .ColdGas import getColdGasMass        
        if verbose:
            print("    cold gas...")
            PROG = Progress(len(datasets))
        dummy = [getColdGasMass(galHDF5Obj,z,name,overwrite=overwrite,\
                                    returnDataset=False,progressObj=PROG) for name in datasets]
        del dummy
