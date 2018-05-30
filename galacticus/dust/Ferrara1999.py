#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import pkg_resources
from .utils import DustProperties,DustClass
from .dustAtlasTables import dustAtlasTable
from .CharlotFall2000 import CharlotFall2000Parameters
from ..io import GalacticusHDF5
from ..Inclination import getInclination
from ..GalacticusErrors import ParseError
from ..Filters import GalacticusFilters
from ..config import *
from ..constants import luminosityAB,luminositySolar
from ..constants import Pi,Parsec,massAtomic,massSolar,massFractionHydrogen



class dustAtlasBase(DustProperties):

    def __init__(self,verbose=False,extrapolateInSize=True,extrapolateInTau=True,\
                     charlotFallParams=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set verbosity
        self.verbose = verbose    
        # Initialise DustProperties class
        super(dustAtlasBase, self).__init__()
        # Create dust table object
        self.dustTable = dustAtlasTable(verbose=self.verbose)
        # Set options whether to extrapolate for various properties
        self.extrapolateInSize = extrapolateInSize
        self.extrapolateInTau = extrapolateInTau
        # Set optical depth parameters for Charlot & Fall (2000) model
        if charlotFallParams is None:
            self.PARAMS = CharlotFall2000Parameters(opticalDepthISMFactor=0.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500.0,wavelengthExponent=0.7)
        else:
            self.PARAMS = charlotFallParams
        return

    def getBulgeSizes(self,component,spheroidMassDistribution,spheroidRadius,diskRadius):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if component.lower() not in ["disk","spheroid"]:
            raise ValueError(funcname+"(): 'component' must be 'spheroid' or 'disk'!")
        if spheroidMassDistribution not in ["hernquist","sersic"]:
            raise ValueError(funcname+"(): 'spheroidMassDistribution' must be 'hernquist' or 'sersic'!")
        np.place(diskRadius,diskRadius<=0.0,1.0)
        # Component bulge sizes
        if component.lower() == "spheroid":
            if fnmatch.fnmatch(spheroidMassDistribution.lower(),"hernquist"):
                sizes = (1.0+np.sqrt(2.0))*spheroidRadius/diskRadius
            elif fnmatch.fnmatch(spheroidMassDistribution.lower(),"sersic"):
                sizes = spheroidRadius/diskRadius
            else:
                pass
            if not self.extrapolateInSize:
                sizeMinimum = self.spheroidAttenuation["size"].min()
                sizeMaximum = self.spheroidAttenuation["size"].max()
                np.place(sizes,sizes<sizeMinimum,sizeMinimum)
                np.place(sizes,sizes>sizeMaximum,sizeMaximum)
        else:
            sizes = None
        return sizes

    def getCloudGasMetalsSurfaceDensity(self,gasMass,gasMetalMass,scaleLength,mcloud=1.0e6,rcloud=16.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        np.place(gasMetalMass,gasMass==0.0,0.0)
        np.place(gasMass,gasMass==0.0,1.0)
        metallicity = gasMetalMass/gasMass
        # Compute surface density
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = metallicity*mcloud/(2.0*Pi*(mega*rcloud)**2)
        np.place(gasMetalsSurfaceDensityCentral,np.isnan(scaleLength),0.0)
        return gasMetalsSurfaceDensityCentral
    
    def getOpticalDepthCentral(self,gasMass,gasMetalMass,scaleLength,mcloud=1.0e6,rcloud=16.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        gasMetalsSurfaceDensityCentral = self.getCloudGasMetalsSurfaceDensity(gasMass,gasMetalMass,scaleLength,mcloud=mcloud,rcloud=rcloud)
        opticalDepthCentral = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)
        if not self.extrapolateInTau:
            if fnmatch.fnmatch(component,"spheroid"):
                tauMinimum = self.spheroidAttenuation["opticalDepth"].min()
                tauMaximum = self.spheroidAttenuation["opticalDepth"].max()
            elif fnmatch.fnmatch(component,"disk"):
                tauMinimum = self.diskAttenuation["opticalDepth"].min()
                tauMaximum = self.diskAttenuation["opticalDepth"].max()
            else:
                raise ValueError(funcname+"(): Value for 'component' not recognised -- must be 'spheroid' or 'disk'!")
            np.place(opticalDepthCentral,opticalDepthCentral<tauMinimum,tauMinimum)
            np.place(opticalDepthCentral,opticalDepthCentral>tauMaximum,tauMaximum)
        return opticalDepthCentral

    def computeOpticalDepthISM(self,gasMetalMass,scaleLength,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        opticalDepthISMFactor = self.PARAMS.opticalDepthISMFactor
        # Compute gas metals central surface density in M_Solar/pc^2
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.PARAMS.wavelengthZeroPoint)**self.PARAMS.wavelengthExponent
        opticalDepth = self.opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = opticalDepthISMFactor*opticalDepth
        return opticalDepthISM

    def computeOpticalDepthClouds(self,gasMass,gasMetalMass,effectiveWavelength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute gas metallicity
        gasMetallicity = self.computeGasMetallicity(np.copy(gasMass),np.copy(gasMetalMass))
        # Compute central optical depth
        wavelengthFactor = (effectiveWavelength/self.PARAMS.wavelengthZeroPoint)**self.PARAMS.wavelengthExponent
        opticalDepthClouds = self.PARAMS.opticalDepthCloudsFactor*np.copy(gasMetallicity)/self.localISMMetallicity/wavelengthFactor
        return opticalDepthClouds

    def computeAttenuationISM(self,component,effectiveWavelength,inclination,spheroidMassDistribution,\
                                  spheroidRadius,diskRadius,gasMass,gasMetalMass,scaleLength):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        sizes = self.getBulgeSizes(component,spheroidMassDistribution,np.copy(spheroidRadius),np.copy(diskRadius))
        gasMetalsSurfaceDensityCentral = self.computeCentralGasMetalsSurfaceDensity(np.copy(gasMetalMass),np.copy(scaleLength))
        opticalDepthCentral = self.getOpticalDepthCentral(np.copy(gasMass),np.copy(gasMetalMass),np.copy(scaleLength),mcloud=1.0e6,rcloud=16.0e-6)
        wavelengths = np.ones_like(inclination)*effectiveWavelength
        attenuationISM = self.dustTable.interpolateDustTable(component,wavelengths,inclination,opticalDepthCentral,bulgeSize=sizes)
        attenuationISM *= np.exp(-self.PARAMS.opticalDepthISMFactor)
        np.place(attenuationISM,diskRadius<=0.0,1.0)
        return attenuationISM



def parseDustAttenuatedLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract dataset name information
    if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
        searchString = "^(?P<component>disk|spheroid)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)(?P<filterName>:[^:]+)?"
    else:
        searchString = "^(?P<component>disk|spheroid)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"        
    searchString = searchString + "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustAtlas(?P<clouds>Clouds)?(?P<options>[^:]+)?)$"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH



class dustAtlas(dustAtlasBase):
    
    def __init__(self,galHDF5Obj,verbose=False,extrapolateInSize=True,extrapolateInTau=True,\
                     charlotFallParams=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise base class class
        super(dustAtlas,self).__init__(verbose=verbose,extrapolateInSize=extrapolateInSize,extrapolateInTau=extrapolateInTau,\
                                           charlotFallParams=charlotFallParams)
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        return


    def getEffectiveWavelength(self,DUST):
        redshift = None
        if DUST.datasetName.group('frame').replace(":","") == "observed":
            redshift = DUST.redshift
        if 'lineName' in DUST.datasetName.groupdict().keys():
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
            raise ValueError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        # Set effective wavelength
        effectiveWavelength = self.getEffectiveWavelength(DUST)
        #  Set inclination
        if "options" in DUST.datasetName.groupdict().keys():
            if DUST.datasetName.group('options') is None:
                inclination = np.copy(np.array(HDF5OUT["nodeData/inclination"]))
            else:
                if fnmatch.fnmatch(DUST.datasetName.group('options').lower(),"faceon"):
                    inclination = np.zeros_like(np.array(HDF5OUT["nodeData/diskRadius"]))
                else:
                    inclination = np.copy(np.array(HDF5OUT["nodeData/inclination"]))
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
        attenuationISM = self.computeAttenuationISM(component,effectiveWavelength,inclination,spheroidMassDistribution,\
                                                        spheroidRadius,diskRadius,gasMass,gasMetalMass,scaleLength)
        np.place(attenuationISM,diskRadius<=0.0,1.0)
        np.place(attenuationISM,attenuationISM==0.0,1.0e-20)
        DUST.opticalDepthISM = -np.log(attenuationISM)
        # Compute attenuation for molecular clouds
        if "clouds" in DUST.datasetName.groupdict().keys():
            if DUST.datasetName.group('clouds') is None:
                DUST.opticalDepthClouds = 0.0
            else:
                DUST.opticalDepthClouds = self.computeOpticalDepthClouds(np.copy(gasMass),np.copy(gasMetalMass),effectiveWavelength)
        else:
            DUST.opticalDepthClouds = 0.0
        return DUST

    def getDustFreeLuminosities(self,DUST):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        # Extract dust free luminosities
        luminosityName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"")
        if "clouds" in DUST.datasetName.groupdict().keys():            
            if DUST.datasetName.group('clouds') is None:
                recentName = None                    
            else:
                if "LineLuminosity" in DUST.datasetName.group(0):
                    recentName = luminosityName
                else:
                    recentName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"recent")
                    if not self.galHDF5Obj.datasetExists(recentName,float(DUST.datasetName.group('redshift'))):
                        raise KeyError(funcname+"(): 'recent' dataset "+recentName+" not found!")
        else:
            recentName = None                    
        luminosity = np.copy(np.array(HDF5OUT["nodeData/"+luminosityName]))
        if recentName is not None:
            recentLuminosity = np.copy(np.array(HDF5OUT["nodeData/"+recentName]))
        else:
            recentLuminosity = np.zeros_like(luminosity)
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
            diskLuminosity = self.getAttenuatedLuminosity(datasetName.replace("total","disk"),z=z,\
                                                              overwrite=overwrite,**kwargs)
            sphereLuminosity = self.getAttenuatedLuminosity(datasetName.replace("total","spheroid"),z=z,\
                                                                overwrite=overwrite,**kwargs)
            DUST.luminosity = np.copy(diskLuminosity) + np.copy(sphereLuminosity)            
            del diskLuminosity,sphereLuminosity
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
