#! /usr/bin/env python


import sys,os,re,fnmatch,copy
import numpy as np
from ..config import *
from ..GalacticusErrors import ParseError
from ..constants import luminosityAB,luminositySolar
from ..utils.progress import Progress
from .utils import DustProperties,DustClass


class CharlotFall2000Parameters(object):

    def __init__(self,opticalDepthISMFactor=None,opticalDepthCloudsFactor=None,\
                     wavelengthZeroPoint=None,wavelengthExponent=None):
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
    
    def reset(self,opticalDepthISMFactor=None,opticalDepthCloudsFactor=None,\
                     wavelengthZeroPoint=None,wavelengthExponent=None):
        self.opticalDepthISMFactor = opticalDepthISMFactor
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        return        

    def default(self):
        self.reset(opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7)
        return



class CharlotFallBase(DustProperties):
    
    def __init__(self,params,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(CharlotFallBase,self).__init__()
        self.PARAMS = params
        self.verbose = verbose        
        return

    def computeOpticalDepthISM(self,gasMetalMass,scaleLength,effectiveWavelength,opticalDepthISMFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if opticalDepthISMFactor is None:
            opticalDepthISMFactor = self.PARAMS.opticalDepthISMFactor
        # Compute gas metals central surface density in M_Solar/pc^2
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.PARAMS.wavelengthZeroPoint)**self.PARAMS.wavelengthExponent
        opticalDepth = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = opticalDepthISMFactor*opticalDepth
        return opticalDepthISM
                           
    def computeOpticalDepthClouds(self,gasMass,gasMetalMass,effectiveWavelength,opticalDepthCloudsFactor=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if opticalDepthCloudsFactor is None:
            opticalDepthCloudsFactor = self.PARAMS.opticalDepthCloudsFactor
        # Compute gas metallicity
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.PARAMS.wavelengthZeroPoint)**self.PARAMS.wavelengthExponent
        opticalDepthClouds = opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        return opticalDepthClouds
    

def parseDustAttenuatedLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract dataset name information
    if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<filterName>:[^:]+)?(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustCharlotFall2000(?P<options>[^:]+)?)$"
    else:
        searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustCharlotFall2000(?P<options>[^:]+)?)$"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH



class CharlotFall2000(CharlotFallBase):
    
    def __init__(self,galHDF5Obj,params,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        super(CharlotFall2000,self).__init__(params,verbose=verbose)
        self.galHDF5Obj = galHDF5Obj
        return

    def getEffectiveWavelength(self,DUST):
        redshift = None
        if DUST.datasetName.group('frame').replace(":","") == "observed":
            redshift = DUST.redshift
        if 'lineName' in DUST.datasetName.re.groupindex.keys():
            name = DUST.datasetName.group('lineName')
            redshift = None
        else:
            name = DUST.datasetName.group('filterName')
        effectiveWavelength = self.effectiveWavelength(name,redshift=redshift,verbose=self.verbose)
        return effectiveWavelength

    def setOpticalDepths(self,DUST,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        # Set component
        component = DUST.datasetName.group('component')
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        # Set effective wavelength
        effectiveWavelength = self.getEffectiveWavelength(DUST)
        #  Set inclination
        if "options" in list(DUST.datasetName.groups()):
            if fnmatch.fnmatch(DUST.datasetName.group('options').lower(),"faceon"):
                inclination = np.zeros_like(np.array(HDF5OUT["nodeData/diskRadius"]))
            else:
                inclination = np.copy(np.array(HDF5OUT["nodeData/inclination"]))
       # Get radii and set scalelengths                                                                                                                                        
        spheroidMassDistribution = self.galHDF5Obj.parameters["spheroidMassDistribution"].lower()
        spheroidRadius = np.copy(np.array(HDF5OUT["nodeData/spheroidRadius"]))
        diskRadius = np.copy(np.array(HDF5OUT["nodeData/diskRadius"]))
        scaleLength = np.copy(np.array(HDF5OUT["nodeData/"+component+"Radius"]))
        # Get gas and metal masses
        gasMass = np.copy(np.array(HDF5OUT["nodeData/"+component+"MassGas"]))
        gasMetalMass = np.copy(np.array(HDF5OUT["nodeData/"+component+"AbundancesGasMetals"]))
        # Compute attenuation for ISM
        DUST.opticalDepthISM = self.computeOpticalDepthISM(gasMetalMass,scaleLength,effectiveWavelength,opticalDepthISMFactor=None)
        # Compute attenuation for molecular clouds
        DUST.opticalDepthClouds = self.computeOpticalDepthClouds(np.copy(gasMass),np.copy(gasMetalMass),effectiveWavelength)
        return DUST

    def getDustFreeLuminosities(self,DUST):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))        
        # Extract dust free luminosities
        luminosityName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"")
        if "LineLuminosity" in DUST.datasetName.group(0):
            recentName = luminosityName
        else:
            recentName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"recent")
            if not self.galHDF5Obj.datasetExists(recentName,float(DUST.datasetName.group('redshift'))):
                raise KeyError(funcname+"(): 'recent' dataset "+recentName+" not found!")
        luminosity = np.copy(np.array(HDF5OUT["nodeData/"+luminosityName]))
        recentLuminosity = np.copy(np.array(HDF5OUT["nodeData/"+recentName]))
        return (luminosity,recentLuminosity)

    def createDustClass(self,datasetName):
        # Create class to store dust optical depths information
        DUST = DustClass()
        DUST.datasetName = parseDustAttenuatedLuminosity(datasetName)        
        # Identify HDF5 output
        DUST.outputName = self.galHDF5Obj.nearestOutputName(float(DUST.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            DUST.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(DUST.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            DUST.redshift = np.ones(ngals,dtype=float)*z        
        return DUST

    def setAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create dust class
        DUST = self.createDustClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        # Check if luminosity already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(DUST.datasetName.group('redshift'))) and not overwrite:
            DUST.luminosity = np.array(HDF5OUT["nodeData/"+datasetName])
            return DUST
        # Compute attenuated luminosity
        if not datasetName.startswith("total"):
            # Compute optical depths
            DUST = self.setOpticalDepths(DUST,**kwargs)
            # Get dust free luminosities
            luminosity,recentLuminosity = self.getDustFreeLuminosities(DUST)
            # Store luminosity
            DUST.luminosity = DUST.attenuate(luminosity,recentLuminosity)
        else:
            diskLum = self.getAttenuatedLuminosity(datasetName.replace("total","disk"),overwrite=overwrite,\
                                                       z=z,**kwargs)
            sphereLum = self.getAttenuatedLuminosity(datasetName.replace("total","spheroid"),overwrite=overwrite,\
                                                         z=z,**kwargs)            
            DUST.luminosity = np.copy(diskLum) + np.copy(sphereLum)
            del sphereLum,diskLum
        return DUST

    def getAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DUST = self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,**kwargs)
        return DUST.luminosity

    def writeLuminosityToFile(self,datasetName,z=None,overwrite=False,**kwargs):
        DUST = self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,**kwargs)
        redshift = float(DUST.datasetName.group('redshift'))
        if not DUST.datasetName.group(0) in self.galHDF5Obj.availableDatasets(redshift) or overwrite:            
            # Select HDF5 output
            HDF5OUT = self.galHDF5Obj.selectOutput(redshift)
            # Add luminosity to file
            self.galHDF5Obj.addDataset(HDF5OUT.name+"/nodeData/",DUST.datasetName.group(0),np.copy(DUST.luminosity))
            # Add appropriate attributes to new dataset
            if fnmatch.fnmatch(DUST.datasetName.group(0),"*LineLuminosity*"):
                attr = {"unitsInSI":luminositySolar}
            else:
                attr = {"unitsInSI":luminosityAB}
            self.galHDF5Obj.addAttributes(out.name+"/nodeData/"+DUST.datasetName.group(0),attr)
        return


