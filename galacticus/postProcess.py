#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from .io import GalacticusHDF5
from .EmissionLines import GalacticusEmissionLine,ContaminateEmissionLine



class processGalacticusHDF5(GalacticusHDF5):
        
    def __init__(self,outfile,verbose=False,overwrite=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initalise HDF5 class
        super(processGalacticusHDF5, self).__init__(outfile,'a')
        # Set verbose and overwrite options
        self.verbose = verbose
        self.overwrite = overwrite
        # Initialise classes for galaxy properties
        self.emissionLines = GalacticusEmissionLine(self)
        self.contaminateLines = ContaminateEmissionLine(self)
        return

    def writeDatasetToFile(self,datasetName,z,dataset,attrs=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.verbose:
            print(funcname+"(): writing dataset "+datasetName+" to file...")
        return

    def processDataset(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        if fnmatch.fnmatch("*LineLuminosity:*"):
            self.processEmissionLine(datasetName,z)
        return
        
    def processDustAttenuation(self,datasetName,z,attrs=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Store name of dataset for output
        outputName = datasetName
        return

    def processEmissionLine(self,datasetName,z):
        # Remove any contaminants and iteratively process line for each individual contaminant
        contaminants = []
        contaminatedName = datasetName
        if ":contam_" in datasetName:
            contaminants = " ".join(fnmatch.filter(datasetName.split(":"),"contam_*"))
            contaminants = contaminants.replace("contam_","").split()
            datasetName = [name if not fnmatch.fnmatch(name,"contam_*") for name in datasetName.split(":")]
            datasetName = ":".join(datasetName)
            lineName = datasetName.split(:)[1]
            dummy = [self.processEmissionLine(datasetName.replace(lineName,contam),z) for contam in contaminants]
        # Remove any dust attenuation and process dust free case
        dust = ""
        if ":dust" in datasetName:
            dust = fnmatch.filter(datasetName.split(":"),"dust*")
            if len(dust)>0:
                dust = dust[0]                
        self.processEmissionLine(datasetName.replace(dust,""),z)
        # Process dust attneuation or write dust-free emission line to file
        if dust is not "":
            # First check if need to process luminosity from recent star formation
            if "CharlotFall" in dust:
                self.processEmissionLine(datasetName.replace(dust,"")+":recent",z)
            # Process dust attenuation
            self.processDustAttenuation(datasetName,z)
        else:
            self.emissionLines.resetLineInformation()
            luminosity = self.emissionLines.getLineLuminosity(datasetName,z=z)
            self.writeDatasetToFile(datasetName,z,luminosity,attrs={"unitsInSI":self.emissionLines.unitsInSI})
        # Process contamination
        if len(contaminants)>0:
            luminosity = self.contaminateLines.getLineLuminosity(contaminatedName,z=z)
            self.writeDatasetToFile(contaminatedName,z,luminosity,attrs={"unitsInSI":self.emissionLines.unitsInSI})
        return


