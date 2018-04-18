#! /usr/bin/env python


import sys,os,re,fnmatch
import numpy as np
from ..config import *
from ..GalacticusErrors import ParseError
from ..constants import luminosityAB,luminositySolar
from ..utils.progress import Progress
from .utils import DustProperties



class CharlotFallBase(DustProperties):
    
    def __init__(self,opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7,\
                     verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(CharlotFallBase,self).__init__()
        self.verbose = verbose        
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

    def resetParameters(self,opticalDepthISMFactor=None,opticalDepthCloudsFactor=None,\
                     wavelengthZeroPoint=None,wavelengthExponent=0.7):
        if opticalDepthISMFactor is not None:
            self.opticalDepthISMFactor = opticalDepthISMFactor
        if opticalDepthCloudsFactor is not None:
            self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        if wavelengthZeroPoint is not None:
            self.wavelengthZeroPoint = wavelengthZeroPoint
        if wavelengthExponent is not None:
            self.wavelengthExponent = wavelengthExponent
        return

    def computeOpticalDepthISM(self,gasMetalMass,scaleLength,effectiveWavelength,opticalDepthISMFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if opticalDepthISMFactor is None:
            opticalDepthISMFactor = self.opticalDepthISMFactor
        # Compute gas metals central surface density in M_Solar/pc^2
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepth = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = opticalDepthISMFactor*opticalDepth
        return opticalDepthISM
                           
    def computeOpticalDepthClouds(self,gasMass,gasMetalMass,effectiveWavelength,opticalDepthCloudsFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if opticalDepthCloudsFactor is None:
            opticalDepthCloudsFactor = self.opticalDepthCloudsFactor
        # Compute gas metallicity
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepthClouds = opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        return opticalDepthClouds
    


class CharlotFall2000(CharlotFallBase):
    
    def __init__(self,galHDF5Obj,opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7,\
                     verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(CharlotFall2000,self).__init__(opticalDepthISMFactor=opticalDepthISMFactor,opticalDepthCloudsFactor=opticalDepthCloudsFactor,\
                                                 wavelengthZeroPoint=wavelengthZeroPoint,wavelengthExponent=wavelengthExponent,\
                                                 verbose=verbose)
        self.datasetName = None
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        # Set variables to store optical depths
        self.opticalDepthISM = None
        self.opticalDepthClouds = None
        self.attenuatedLuminosiy = None
        return

    def resetHDF5Output(self):
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        return

    def reset(self):
        self.resetAttenuationInformation()
        return

    def resetAttenuationInformation(self):
        self.opticalDepthISM = None
        self.opticalDepthClouds = None
        self.attenuatedLuminosity = None
        return

    def setHDF5Output(self,z):
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
            searchString = "^(?P<component>disk|spheroid)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<filterName>:[^:]+)?(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustCharlotFall2000(?P<options>[^:]+)?)$"
        else:
            searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustCharlotFall2000(?P<options>[^:]+)?)$"
        self.datasetName = re.search(searchString,datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return

    def getEffectiveWavelength(self):
        redshift = None
        if self.datasetName.group('frame').replace(":","") == "observed":
            redshift = float(self.redshift)
        if 'lineName' in self.datasetName.re.groupindex.keys():
            name = self.datasetName.group('lineName')
            redshift = None
        else:
            name = self.datasetName.group('filterName')
        effectiveWavelength = self.effectiveWavelength(name,redshift=redshift,verbose=self.verbose)
        return effectiveWavelength


    def setOpticalDepths(self,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Reset attenuation information
        self.resetAttenuationInformation()
        # Set component
        component = self.datasetName.group('component')
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        # Set effective wavelength
        effectiveWavelength = self.getEffectiveWavelength()
        #  Set inclination
        if "options" in list(self.datasetName.groups()):
            if fnmatch.fnmatch(self.datasetName.group('options').lower(),"faceon"):
                inclination = np.zeros_like(np.array(self.hdf5Output["nodeData/diskRadius"]))
            else:
                inclination = np.copy(np.array(self.hdf5Output["nodeData/inclination"]))
       # Get radii and set scalelengths                                                                                                                                        
        spheroidMassDistribution = self.galHDF5Obj.parameters["spheroidMassDistribution"].lower()
        spheroidRadius = np.copy(np.array(self.hdf5Output["nodeData/spheroidRadius"]))
        diskRadius = np.copy(np.array(self.hdf5Output["nodeData/diskRadius"]))
        scaleLength = np.copy(np.array(self.hdf5Output["nodeData/"+component+"Radius"]))
        # Get gas and metal masses
        gasMass = np.copy(np.array(self.hdf5Output["nodeData/"+component+"MassGas"]))
        gasMetalMass = np.copy(np.array(self.hdf5Output["nodeData/"+component+"AbundancesGasMetals"]))
        # Compute attenuation for ISM
        self.opticalDepthISM = self.computeOpticalDepthISM(gasMetalMass,scaleLength,effectiveWavelength,opticalDepthISMFactor=None)
        # Compute attenuation for molecular clouds
        self.opticalDepthClouds = self.computeOpticalDepthClouds(np.copy(gasMass),np.copy(gasMetalMass),effectiveWavelength)
        return

    def applyAttenuation(self,luminosity,recentLuminosity):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        attenuationISM = np.exp(-self.opticalDepthISM)
        attenuationClouds = np.exp(-self.opticalDepthClouds)
        result = ((luminosity-recentLuminosity) + recentLuminosity*attenuationClouds)*attenuationISM
        return result

    def setAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
       # Check HDF5 snapshot specified
        if z is not None:
            self.setHDF5Output(z)
        else:
            if self.hdf5Output is None:
                z = self.datasetName.group('redshift')
                if z is None:
                    errMsg = funcname+"(): no HDF5 output specified. Either specify the redshift "+\
                        "of the output or include the redshift in the dataset name."
                    raise RunTimeError(errMsg)
                self.setHDF5Output(z)
        # Set datasetName
        self.setDatasetName(datasetName)
        # Compute attenuated luminosity
        if self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) and not overwrite:
            self.attenuatedLuminosity = np.array(self.hdf5Output["nodeData/"+datasetName])
            return
        # Compute optical depths
        self.setOpticalDepths(**kwargs)
        # Extract dust free luminosities
        luminosityName = datasetName.replace(self.datasetName.group('dust'),"")
        if "LineLuminosity" in self.datasetName.group(0):
            recentName = luminosityName
        else:
            recentName = datasetName.replace(self.datasetName.group('dust'),"recent")
        luminosity = np.copy(np.array(self.hdf5Output["nodeData/"+luminosityName]))
        recentLuminosity = np.copy(np.array(self.hdf5Output["nodeData/"+recentName]))
        self.attenuatedLuminosity = self.applyAttenuation(luminosity,recentLuminosity)
        return

    def getAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName.startswith("total"):
            luminosity = self.getAttenuatedLuminosity(datasetName.replace("total","disk"),overwrite=overwrite,z=z,**kwargs) + \
                self.getAttenuatedLuminosity(datasetName.replace("total","spheroid"),overwrite=overwrite,z=z,**kwargs)
        else:
            self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,**kwargs)
            luminosity = self.attenuatedLuminosity
        return luminosity


    def writeLuminosityToFile(self,overwrite=False):
        if not self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) or overwrite:
            out = self.galHDF5Obj.selectOutput(self.redshift)
            # Add luminosity to file
            self.galHDF5Obj.addDataset(out.name+"/nodeData/",self.datasetName.group(0),np.copy(self.attenuatedLuminosity))
            # Add appropriate attributes to new dataset
            if fnmatch.fnmatch(self.datasetName.group(0),"*LineLuminosity*"):
                attr = {"unitsInSI":luminositySolar}
            else:
                attr = {"unitsInSI":luminosityAB}
            self.galHDF5Obj.addAttributes(out.name+"/nodeData/"+self.datasetName.group(0),attr)
        return


