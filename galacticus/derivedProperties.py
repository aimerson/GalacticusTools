#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .utils.progress import Progress
from .EmissionLines import GalacticusEmissionLines

class derivedProperties(object):
    
    def __init__(self,galHDF5Obj,verbose=True):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.EmissionLines = None
        self._verbose = verbose
        return

    def addProperties(self,z,derivedDatasets,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name

        PROG = None
        if self._verbose:
            print(funcname+"(): processing derived datasets:")       
            
        # Stellar masses
        datasets = fnmatch.filter(derivedDatasets,"*MassStellar")
        if len(datasets)>0: 
            from .Stars import getStellarMass
            if self._verbose:
                print("    stellar mass...")
                PROG = Progress(len(datasets))
            dummy = [getStellarMass(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                        returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy    
        # Star formation rates
        datasets = fnmatch.filter(derivedDatasets,"*StarFormationRate")
        if len(datasets)>0: 
            from .Stars import getStarFormationRate
            if self._verbose:
                print("    star formation rate...")
                PROG = Progress(len(datasets))
            dummy = [getStarFormationRate(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                              returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy
        # Stellar luminosities 
        datasets = fnmatch.filter(derivedDatasets,"*LuminositiesStellar:*")
        datasets = list(set(datasets).difference(fnmatch.filter(datasets,"*:dust*")))
        if len(datasets)>0: 
            from .Luminosities import getLuminosity
            if self._verbose:
                print("    stellar luminosity...")
                PROG = Progress(len(datasets))
            dummy = [getLuminosity(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                       returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy
        
        # Emission line luminosities
        datasets = fnmatch.filter(derivedDatasets,"*LineLuminosity:*")
        if len(datasets)>0:         
            if self.EmissionLines is None:
                self.EmissionLines = GalacticusEmissionLines()
            if self._verbose:
                print("    emission line luminosities...")
                PROG = Progress(len(datasets))
            dummy = [self.EmissionLines.getLineLuminosity(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                                              returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy
        # Emission line equivalent widths
        datasets = fnmatch.filter(derivedDatasets,"*EquivalentWidth:*")
        if len(datasets)>0:         
            if self.EmissionLines is None:
                self.EmissionLines = GalacticusEmissionLines()
            if self._verbose:
                print("    emission line equivalent widths...")
                PROG = Progress(len(datasets))
            dummy = [self.EmissionLines.getEquivalentWidth(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                                               returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy
        # Cold gas
        datasets = fnmatch.filter(derivedDatasets,"*MassGas") + fnmatch.filter(derivedDatasets,"*MassColdGas")
        if len(datasets)>0: 
            from .ColdGas import getColdGasMass        
            if self._verbose:
                print("    cold gas...")
                PROG = Progress(len(datasets))
            dummy = [getColdGasMass(self.galHDF5Obj,z,name,overwrite=overwrite,\
                                        returnDataset=False,progressObj=PROG) for name in datasets]
            del dummy

        return
