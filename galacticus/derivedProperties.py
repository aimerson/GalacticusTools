#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .utils.progress import Progress

from .Stars import getStellarMass,getStarFormationRate
from .ColdGas import getColdGasMass
from .Inclination import getInclination
from .EmissionLines import GalacticusEmissionLines
from .dust.utils import getDustFreeName
from .dust.Ferrara2000 import dustAtlas
from .dust.CharlotFall2000 import CharlotFall2000




class derivedProperties(object):
    
    def __init__(self,galHDF5Obj,dustFerrara200=None,dustCharlotFall2000=None,verbose=True):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        # Emission lines class
        self.EmissionLines = None
        # Dust classes
        self.dustFerrara2000 = dustFerrara2000
        self.dustCharlotFall2000 = dustCharlotFall2000
        # Set verbosity
        self._verbose = verbose
        return

    def addDatasets(self,z,derivedDatasets,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        PROG = len(derivedDatasets)
        if self._verbose:
            print(funcname+"(): processing derived datasets:")       
        dummy = [self.addDataset(datasetName,z,overwrite=overwrite,progObj=PROG) \
                     for datasetName in derivedDatasets]

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
        return


    def addDataset(self,datasetName,z,overwrite=False,progObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Control to skip checking for dust attenuation for this dataset
        skipDust = False
        # Stellar mass
        if fnmatch.fnmatch(datasetName,"*MassStellar"):
            getStellarMass(self.galHDF5Obj,z,datasetName,overwrite=overwrite,returnDataset=False)            
        # Star formation rate
        if fnmatch.fnmatch(datasetName,"*StarFormationRate"):
            getStarFormationRate(self.galHDF5Obj,z,datasetName,overwrite=overwrite,returnDataset=False)
        # Cold gas
        if fnmatch.fnmatch(datasetName,"*MassGas") or fnmatch.fnmatch(datasetName,"*MassColdGas"):
            getColdGasMass(self.galHDF5Obj,z,datasetName,overwrite=overwrite,returnDataset=False)
        # Inclination
        if fnmatch.fnmatch(datasetName,"inclination"):
            getInclination(self.galHDF5Obj,z,overwrite=overwrite,returnDataset=False)
        # Stellar luminosity
        if fnmatch.fname(datasetName,"*LuminositiesStellar:*"):
            getLuminosity(self.galHDF5Obj,z,getDustFreeName(datasetName),overwrite=overwrite,returnDataset=False)        
        # Emission line luminosities
        if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
            if self.EmissionLines is None:
                self.EmissionLines = GalacticusEmissionLines()
            if fnmatch.fnmatch(datasetName,"total"):
                diskName = getDustFreeName(datasetName.replace("total","disk"))
                self.EmissionLines.getLineLuminosity(self.galHDF5Obj,z,diskName,overwrite=overwrite,returnDataset=False)
                spheroidName = getDustFreeName(datasetName.replace("total","spheroid"))
                self.EmissionLines.getLineLuminosity(self.galHDF5Obj,z,spheroidName,overwrite=overwrite,returnDataset=False)
            else:
                self.EmissionLines.getLineLuminosity(self.galHDF5Obj,z,getDustFreeName(datasetName),overwrite=overwrite,\
                                                         returnDataset=False)
        # Emission line equivalent widths
        if fnmatch.fnmatch(datasetName,"*EquivalentWidth:*"):
            skipDust = True
            if self.EmissionLines is None:
                self.EmissionLines = GalacticusEmissionLines()
            self.EmissionLines.getEquivalentWidth(self.galHDF5Obj,z,datasetName,overwrite=overwrite,\
                                                      returnDataset=False)
        # Dust attenuation
        if fnmatch.fnmatch(datasetName,"*:dust*") and not skipDust:
            # Charlot & Fall (2000)
            if fnmatch.fnmatch(datasetName,"*:dustCharlotFall2000*"):
                if self.dustCharlotFall2000 is None:
                    self.dustCharlotFall2000 = CharlotFall2000()
                self.dustCharlotFall2000.attenuate(self.galHDF5Obj,z,datasetName,overwrite=overwrite,returnDataset=False)
            # Ferrara et al. (2000)
            if fnmatch.fnmatch(datasetName,"*:dustAtlas*"):
                if self.dustFerrara2000 is None:
                    self.dustFerrara2000 = dustAtlas()
                self.dustFerrara2000.attenuate(self.galHDF5Obj,z,datasetName,overwrite=False,returnDataset=False,\
                                                   extrapolateInSize=True,extrapolateInTau=True)
        # Update progress and return
        if progObj is not None:
            progObj.increment()
            progObj.print_status_line()
        return
