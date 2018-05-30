#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .EmissionLines import GalacticusEmissionLine,ContaminateEmissionLine
from .IonizingContinuua import IonizingContinuua
from .Luminosities import StellarLuminosities
from .Inclination import getInclination
from .Stars import GalacticusStellarMass,GalacticusStarFormationRate
from .dust.Ferrara1999 import dustAtlas
from .dust.CharlotFall2000 import CharlotFall2000
from .dust.screens import dustScreen


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
        self.ionizingContinuua = IonizingContinuua(self)
        self.contaminateLines = ContaminateEmissionLine(self)
        self.stellarLuminosities = StellarLuminosities(self)
        self.stellarMass = GalacticusStellarMass(self)
        self.starFormationRate = GalacticusStarFormationRate(self)
        self.DUSTATLAS = dustAtlas(self)
        self.DUSTCF2000 = CharlotFall2000(self)
        self.DUSTSCREEN = dustScreen(self)
        return

    def writeDatasetToFile(self,datasetName,z,dataset,attrs=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        if self.verbose:     
            print(funcname+"(): writing dataset "+datasetName+" to file...")
        OUT = self.selectOutput(z)
        self.addDataset(OUT.name+"/nodeData",datasetName,dataset,append=False,overwrite=self.overwrite,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6)
        if attrs is not None:
            self.addAttributes(OUT.name+"/nodeData/"+datasetName,attrs)
        return

    def processDataset(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
            self.processEmissionLine(datasetName,z)
        if fnmatch.fnmatch(datasetName,"*LuminositiesStellar:*"):
            self.processStellarLuminosity(datasetName,z)            
        if fnmatch.fnmatch(datasetName,"bulgeToTotalLuminosities:*"):
            self.processBulgeToTotalRatio(datasetName,z)            
        if fnmatch.fnmatch(datasetName,"*MassStellar"):
            self.processStellarMass(datasetName,z)            
        if fnmatch.fnmatch(datasetName,"*StarFormationRate"):
            self.processStarFormationRate(datasetName,z)            
        if fnmatch.fnmatch(datasetName,"inclination"):
            self.processInclination(datasetName,z)            
        if fnmatch.fnmatch(datasetName,"*ContinuumLuminosity*"):
            self.processIonizingContinuum(datasetName,z)
        return

    def processInclination(self,datasetName,z):
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        inclination = getInclination(self,z)
        self.writeDatasetToFile(datasetName,z,inclination)
        return
                
    def processStellarMass(self,datasetName,z):
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        if datasetName.startswith("total"):
            # Extract stellar mass
            stellarMass = self.stellarMass.getStellarMass(datasetName,z)
            # Write to file 
            self.writeDatasetToFile(datasetName,z,stellarMass,attrs={"unitsInSI":self.stellarMass.unitsInSI})
        return

    def processStarFormationRate(self,datasetName,z):
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        if datasetName.startswith("total"):
            # Extract star formation rate
            stellarMass = self.starFormationRate.getStarFormationRate(datasetName,z)
            # Write to file 
            self.writeDatasetToFile(datasetName,z,stellarMass,attrs={"unitsInSI":self.starFormationRate.unitsInSI})
        return

    def processIonizingContinuum(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # Check if require dust attenuation
        if "dust" in datasetName:                        
            luminosityName = self.ionizingContinuua.getStellarLuminosityName(datasetName)
            self.processDataset(luminosityName,z)
        # Get luminosity
        luminosity = self.ionizingContinuua.getIonizingLuminosity(datasetName,z=z)
        # Write to file
        self.writeDatasetToFile(datasetName,z,luminosity)
        return

    def processEmissionLine(self,datasetName,z):
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # If total, process disk and spheroid components
        if datasetName.startswith("total"):
            self.processEmissionLine(datasetName.replace("total","disk"),z)
            self.processEmissionLine(datasetName.replace("total","spheroid"),z)
        # Remove any contaminants and iteratively process line for each individual contaminant
        contaminants = []
        contaminatedName = datasetName
        if ":contam_" in datasetName:
            contaminants = " ".join(fnmatch.filter(datasetName.split(":"),"contam_*"))
            contaminants = contaminants.replace("contam_","").split()
            datasetName = [name for name in datasetName.split(":") if not fnmatch.fnmatch(name,"contam_*")]
            datasetName = ":".join(datasetName)
            lineName = datasetName.split(":")[1]
            dummy = [self.processEmissionLine(datasetName.replace(lineName,contam),z) for contam in contaminants]
        # Remove any dust attenuation and process dust free case
        dust = ""
        if ":dust" in datasetName:
            dust = fnmatch.filter(datasetName.split(":"),"dust*")
            if len(dust)>0:
                dust = dust[0]                
            self.processEmissionLine(datasetName.replace(":"+dust,""),z)
        # Process dust attenuation or write dust-free emission line to file
        if dust is not "":
            # Process dust attenuation
            self.processDustAttenuation(datasetName,z)
        else:
            #self.emissionLines.resetLineInformation()
            EMLINE = self.emissionLines.getLineLuminosity(datasetName,z=z)            
            self.writeDatasetToFile(datasetName,z,EMLINE.luminosity,attrs={"unitsInSI":self.emissionLines.unitsInSI})
        # Process contamination
        if len(contaminants)>0:
            luminosity = self.contaminateLines.getLineLuminosity(contaminatedName,z=z)
            self.writeDatasetToFile(contaminatedName,z,luminosity,attrs={"unitsInSI":self.emissionLines.unitsInSI})
        return

    def processStellarLuminosity(self,datasetName,z):
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # Remove any dust attenuation and process dust free case
        dust = ""
        if ":dust" in datasetName:
            dust = fnmatch.filter(datasetName.split(":"),"dust*")
            if len(dust)>0:
                dust = dust[0]                
            self.processStellarLuminosity(datasetName.replace(dust,""),z)
        # Process dust attenuation or write dust-free emission line to file
        if dust is not "":
            # First check if need to process luminosity from recent star formation
            if "CharlotFall" in dust:
                self.processStellarLuminosity(datasetName.replace(dust,"")+":recent",z)
            # Process dust attenuation
            self.processDustAttenuation(datasetName,z)
        else:
            luminosity = self.stellarLuminosities.getLuminosity(datasetName,z=z)
            self.writeDatasetToFile(datasetName,z,luminosity,attrs={"unitsInSI":self.stellarLuminosities.unitsInSI})
        return

    def processBulgeToTotalRatio(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # Construct names of separate stellar luminosities to process
        spheroidName = datasetName.replace("bulgeToTotalLuminosities","spheroidLuminositiesStellar")
        totalName = datasetName.replace("bulgeToTotalLuminosities","totalLuminositiesStellar")
        # Process separate components        
        self.processStellarLuminosity(spheroidName,z)
        self.processStellarLuminosity(totalName,z)
        # Extract ratio and write to file
        ratio = self.stellarLuminosities.getBulgeToTotal(datasetName,z)
        self.writeDatasetToFile(datasetName,z,ratio)
        return

    def processDustAttenuation(self,datasetName,z,attrs=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # Get dust attenuated luminosity
        if fnmatch.fnmatch(datasetName,"*dustAtlas*"):
            self.processInclination("inclination",z)
            DUST = self.DUSTATLAS.getAttenuatedLuminosity(datasetName,z=z,**kwargs)
        elif fnmatch.fnmatch(datasetName,"*dustCharlotFall*"):
            DUST = self.DUSTCF2000.getAttenuatedLuminosity(datasetName,z=z)
        elif fnmatch.fnmatch(datasetName,"*dustScreen*"):
            DUST = self.DUSTSCREEN.getAttenuatedLuminosity(datasetName,z=z,Rv=None)
        # Write to file
        self.writeDatasetToFile(DUST.datasetName.group(0),z,DUST.attenuatedLuminosity)
        return
        
    def processIonizingContinuum(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.datasetExists(datasetName,z) and not self.overwrite:
            return
        # Check if require dust attenuation
        if "dust" in datasetName:                        
            luminosityName = self.ionizingContinuua.getStellarLuminosityName(datasetName)
            self.processDataset(luminosityName,z)
        # Get luminosity
        luminosity = self.ionizingContinuua.getIonizingLuminosity(datasetName,z=z)
        # Write to file
        self.writeDatasetToFile(datasetName,z,luminosity)
        return
