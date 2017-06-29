#! /usr/bin/env python


import sys,os,re,fnmatch
import numpy as np
from ..config import *
from ..GalacticusErrors import ParseError
from ..constants import luminosityAB,luminositySolar
from ..utils.progress import Progress
from .utils import DustProperties




class CharlotFall2000(DustProperties):
    
    def __init__(self,opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7,\
                     verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(CharlotFall2000,self).__init__()
        self._verbose = verbose        
        # Specify constants in extinction model 
        # -- "wavelengthExponent" is set to the value of 0.7 found by Charlot & Fall (2000). 
        # -- "opticalDepthCloudsFactor" is set to unity, such that in gas with Solar metallicity the cloud optical depth will be 1.
        # -- "opticalDepthISMFactor" is set to 1.0 such that we reproduce the standard (Bohlin et al 1978) relation between visual
        #     extinction and column density in the local ISM (essentially solar metallicity).
        self.opticalDepthISMFactor = opticalDepthISMFactor
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        return


    def computeOpticalDepthISM(self,gasMetalMass,scaleLength,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Compute gas metals central surface density in M_Solar/pc^2
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepth = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = self.opticalDepthISMFactor*opticalDepth
        return opticalDepthISM
                           
    def computeOpticalDepthClouds(self,gasMass,gasMetalMass,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Compute gas metallicity
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepthClouds = self.opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        return opticalDepthClouds
    
    def applyAttenuation(self,luminosity,recentLuminosity,opticalDepthISM,opticalDepthClouds):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        attenuationISM = np.exp(-opticalDepthISM)
        attenuationClouds = np.exp(-opticalDepthClouds)
        result = ((luminosity-recentLuminosity) + recentLuminosity*attenuationClouds)*attenuationISM
        return result


    def computeAttenuation(self,galHDF5Obj,z,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check is a luminosity for attenuation
        MATCH = re.search(r"^(disk|spheroid)([^:]+):([^:]+):([^:]+):z([\d\.]+)(:contam_[^:]+)?:dustCharlotFall2000([^:]+)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract dataset information
        component = MATCH.group(1)
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        luminosityOrOpticalDepth = MATCH.group(2)
        if luminosityOrOpticalDepth not in ["CentralOpticalDepthISM","CentralOpticalDepthClouds","LuminositiesStellar","LineLuminosity"]:
            raise ParseError(funcname+"(): Cannot identify type of luminosity or optical depth from '"+datasetName+"'!")
        filter = MATCH.group(3)
        frame = MATCH.group(4)
        redshift = MATCH.group(5)
        if self._verbose:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)
        contamination = MATCH.group(6)
        if contamination is None:
            contamination = ""
        dustExtension = ":dustCharlotFall2000"
        dustOption = MATCH.group(7)
        faceOn = False
        includeClouds = True
        if dustOption is not None:
            dustExtension = dustExtension + dustOption
            if "noclouds" in dustOption.lower():
                includeClouds = False
                opticalDepthCloudsFactor = 0.0
            else:
                includeClouds = True
                opticalDepthCloudsFactor = self.opticalDepthCloudsFactor
            if "faceon" in dustOption.lower():
                faceOn = True
            else:
                faceOn = False
        # Get name for luminosity from recent star formation
        luminosityDataset = datasetName.replace(dustExtension,"")
        if fnmatch.fnmatch(luminosityOrOpticalDepth,"LineLuminosity"):
            recentLuminosityDataset = luminosityDataset    
        else:
            recentLuminosityDataset = luminosityDataset+":recent"    
            if not recentLuminosityDataset in list(map(str,out["nodeData"].keys())):
                raise IOError(funcname+"(): Missing luminosity for recent star formation for filter '"+luminosityDataset+"'!")
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift+contamination
        # Compute effective wavelength for filter/line
        if frame == "observed":
            effectiveWavelength = self.effectiveWavelength(filter,redshift=float(redshift),verbose=self._verbose)
        else:
            effectiveWavelength = self.effectiveWavelength(filter,redshift=None,verbose=self._verbose)
        # Load required galaxy properties
        gasMass = np.array(out["nodeData/"+component+"MassGas"])
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        scaleLength = np.array(out["nodeData/"+component+"Radius"])
        # Compute central optical depths
        opticalDepthISM = np.copy(self.computeOpticalDepthISM(gasMetalMass,scaleLength,effectiveWavelength))
        opticalDepthClouds = np.copy(self.computeOpticalDepthClouds(gasMass,gasMetalMass,effectiveWavelength))
        del gasMass,gasMetalMass,scaleLength        
        # Compute the attenutations of ISM and clouds
        attenuationISM = np.exp(-opticalDepthISM)
        attenuationClouds = np.exp(-opticalDepthClouds)
        # Apply attenuations to dataset or return optical depths
        if fnmatch.fnmatch(luminosityOrOpticalDepth,"LuminositiesStellar"):
            # i) stellar luminosities
            result = self.applyAttenuation(np.array(out["nodeData/"+luminosityDataset]),np.array(out["nodeData/"+recentLuminosityDataset]),\
                                               opticalDepthISM,opticalDepthClouds)
            #result = np.array(out["nodeData/"+luminosityDataset]) - np.array(out["nodeData/"+recentLuminosityDataset]) 
            #result += np.array(out["nodeData/"+recentLuminosityDataset])*attenuationClouds
            #result *= attenuationISM
        elif fnmatch.fnmatch(luminosityOrOpticalDepth,"LineLuminosity"):
            # ii) emission lines
            result = self.applyAttenuation(np.array(out["nodeData/"+luminosityDataset]),np.array(out["nodeData/"+luminosityDataset]),\
                                               opticalDepthISM,opticalDepthClouds)
            #result = np.array(out["nodeData/"+luminosityDataset])*attenuationClouds*attenuationISM
        else:
            # iii) return appropriate optical depth
            if fnmatch.fnmatch(luminosityOrOpticalDepth,"CentralOpticalDepthISM"):
                result = np.copy(opticalDepthISM)
            elif fnmatch.fnmatch(luminosityOrOpticalDepth,"CentralOpticalDepthClouds"):
                result = np.copy(opticalDepthClouds)
        # Return resulting attenuated dataset
        return result


    def attenuate(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check if dust attenuated luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            if progressObj is not None:
                progressObj.increment()
                progressObj.print_status_line()
            if returnDataset:
                out = galHDF5Obj.selectOutput(z)
                return np.array(out["nodeData/"+datasetName])
            else:
                return
        # Check if a total luminosity or disk/spheroid luminosity
        if datasetName.startswith("total"):
            diskResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","disk"),overwrite=False,returnDataset=True)
            spheroidResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","spheroid"),overwrite=False,returnDataset=True)
            result = np.copy(diskResult) + np.copy(spheroidResult)
            del diskResult,spheroidResult
        else:
            result = self.computeAttenuation(galHDF5Obj,z,datasetName)
        # Write property to file and return result
        out = galHDF5Obj.selectOutput(z)
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,result)
        attr = None
        if fnmatch.fnmatch(datasetName,"*LuminositiesStellar*"):
            attr = {"unitsInSI":luminosityAB}
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):
            attr = {"unitsInSI":luminositySolar}
        if attr is not None:
            galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return result
        return






