#! /usr/bin/env python

import fnmatch
import numpy as np
from ..config import *
from ..GalacticusErrors import ParseError
from ..EmissionLines import emissionLines
from ..Filters import GalacticusFilters
from ..constants import Pi,Parsec,massAtomic,massSolar,massFractionHydrogen



class CharlotFall2000(object):
    
    def __init__(self,opticalDepthISMFactor=1.0,opticalDepthCloudsFactor=1.0,\
                     wavelengthZeroPoint=5500,wavelengthExponent=0.7,\
                     verbose=False,debug=False):
        self.verbose = verbose        
        self.debug = debug        
        # Specify constants in extinction model 
        # -- "wavelengthExponent" is set to the value of 0.7 found by Charlot & Fall (2000). 
        # -- "opticalDepthCloudsFactor" is set to unity, such that in gas with Solar metallicity the cloud optical depth will be 1.
        # -- "opticalDepthISMFactor" is set to 1.0 such that we reproduce the standard (Bohlin et al 1978) relation between visual
        #     extinction and column density in the local ISM (essentially solar metallicity).
        self.opticalDepthISMFactor = opticalDepthISMFactor
        self.opticalDepthCloudsFactor = opticalDepthCloudsFactor
        self.wavelengthZeroPoint = wavelengthZeroPoint
        self.wavelengthExponent = wavelengthExponent
        # Initialise classes for emission lines and filters
        self.emissionLinesClass = emissionLines()
        self.filtersDatabase = GalacticusFilters()
        return


    def attenuate(self,galHDF5Obj,z,datasetName,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if dust attenuated luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])        
        ####################################################################
        # Compute attenuation 
        if self.debug:
            print(funcname+"(): Processing dataset '"+datasetName+"'")                    
        if not fnmatch.fnmatch(datasetName,"*:dustCharlotFall2000*"):
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Check whether to include attenuation by molecular clouds and get name of unattenuated dataset
        includeClouds = True
        dustExtension = ":dustCharlotFall2000"
        if "noClouds" in datasetName:
            opticalDepthCloudsFactor = 0.0
            dustExtension = dustExtension + "noClouds"
        else:
            opticalDepthCloudsFactor = self.opticalDepthCloudsFactor
        luminosityDataset = datasetName.replace(dustExtension,"")
        # Get name for luminosity from recent star formation
        recentLuminosityDataset = luminosityDataset+":recent"
        if not recentLuminosityDataset in out["nodeData"].keys():
            raise IOError(funcname+"(): Missing luminosity for recent star formation for filter '"+luminosityDataset+"'!")
        # Get component of dataset (disk or spheroid)
        if luminosityDataset.startswith("disk"):
            component = "disk"
        elif luminosityDataset.startswith("spheroid"):
            component = "spheroid"
        else:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")        
        # Extract dataset information        
        datasetInfo = luminosityDataset.split(":")
        opticalDepthOrLuminosity = datasetInfo[0].replace(component,"")
        if opticalDepthOrLuminosity == "LineLuminosity":
            emissionLineFlag = True
        else:
            emissionLineFlag = False
        filter = datasetInfo[1]
        frame = datasetInfo[2]
        redshift = datasetInfo[3].replace("z","")
        if self.debug:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)            
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift
        # Compute effective wavelength for filter/line
        if emissionLineFlag:
            # i) emission lines
            effectiveWavelength = self.emissionLinesClass.getWavelength(filter)
            if debug:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Emission line filter information:\n        "+infoLine)            
        else:
            # ii) photometric filters
            effectiveWavelength = self.filtersDatabase.getEffectiveWavelength(filter,verbose=self.debug)
            if frame == "observed":
                effectiveWavelength /= (1.0+float(redshift))
            if self.debug:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(filter,effectiveWavelength)
                print(funcname+"(): Photometric filter information:\n        "+infoLine)            
        # Specify required constants
        localISMMetallicity = 0.02  # ... Metallicity in the local ISM.
        AV_to_EBV = 3.10            # ... (A_V/E(B-V); Savage & Mathis 1979)                                                                                                                           
        NH_to_EBV = 5.8e21          # ... (N_H/E(B-V); atoms/cm^2/mag; Savage & Mathis 1979)  
        opticalDepthToMagnitudes = 2.5*np.log10(np.exp(1.0)) # Conversion factor from optical depth to magnitudes of extinction.
        hecto = 1.0e2
        opticalDepthNormalization = (1.0/opticalDepthToMagnitudes)*(AV_to_EBV/NH_to_EBV)
        opticalDepthNormalization *= (massFractionHydrogen/massAtomic)*(massSolar/(Parsec*hecto)**2)/localISMMetallicity
        # Compute central surface density in M_Solar/pc^2
        gasMass = np.array(out["nodeData/"+component+"MassGas"])
        gasMetalMass = np.array(out["nodeData/"+component+"AbundancesGasMetals"])
        noGasMetals = gasMetalMass <= 0.0
        scaleLength = np.array(out["nodeData/"+component+"Radius"])
        notPresent = scaleLength <= 0.0
        gasMetallicity = gasMetalMass/gasMass
        np.place(gasMetallicity,noGasMetals,0.0)
        mega = 1.0e6
        gasMetalsSurfaceDensityCentral = gasMetalMass/(2.0*Pi*(mega*scaleLength)**2)
        np.place(gasMetalsSurfaceDensityCentral,notPresent,0.0)
        # Compute central optical depths
        wavelengthFactor = (effectiveWavelength/self.wavelengthZeroPoint)**self.wavelengthExponent
        opticalDepth = opticalDepthNormalization*np.copy(gasMetalsSurfaceDensityCentral)/wavelengthFactor
        opticalDepthISM = self.opticalDepthISMFactor*opticalDepth
        opticalDepthClouds = opticalDepthCloudsFactor*np.copy(gasMetallicity)/localISMMetallicity/wavelengthFactor
        del gasMetalsSurfaceDensityCentral,gasMetallicity,gasMass,gasMetalMass,noGasMetals,scaleLength,noPresent
        # Compute the attenutations of ISM and clouds
        attenuationsISM = np.exp(-opticalDepthISM)
        attenuationsClouds = np.exp(-opticalDepthClouds)
        # Apply attenuations to dataset or return optical depths
        if fnmatch.fnmatch(opticalDepthOrLuminosity,"LuminositiesStellar"):
            # i) stellar luminosities
            result = np.array(out["nodeData/"+luminosityDataSet]) - np.array(out["nodeData/"+recentLuminosityDataSet]) 
            result += np.array(out["nodeData/"+recentLuminosityDataSet])*attenuationClouds
            result *= attenuationISM
        elif fnmatch.fnmatch(opticalDepthOrLuminosity,"LineLuminosity"):
            # ii) emission lines
            result = np.array(out["nodeData/"+luminosityDataSet])*attenuationClouds*attenuationISM
        else:
            # iii) return appropriate optical depth
            if fnmatch.fnmatch(opticalDepthOrLuminosity,"CentralOpticalDepthISM"):
                result = np.copy(opticalDepthISM)
            elif fnmatch.fnmatch(opticalDepthOrLuminosity,"CentralOpticalDepthClouds"):
                result = np.copy(opticalDepthClouds)
        ####################################################################
        # Write property to file and return result
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,result)
        return result


