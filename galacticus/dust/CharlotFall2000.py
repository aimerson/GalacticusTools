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



       
class depracated(object):
    
    
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






