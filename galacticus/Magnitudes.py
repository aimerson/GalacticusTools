#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
from .utils.progress import Progress
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .Luminosities import LuminosityClass
from .StellarLuminosities import StellarLuminosities
from .Filters import GalacticusFilters

class MagnitudeClass(LuminosityClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,\
                     luminosity=None):
        super(MagnitudeClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName,\
                                                luminosity=luminosity)
        self.magnitude = None
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.luminosity = None
        self.magnitude = None
        return


def parseMagitude(datasetName):
    funcname = sys._getframe().f_code.co_name
    #MATCH = re.search(r"^magnitude([^:]+):([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:vega|:AB)?",datasetName)
    searchString = "^(?P<component>disk|spheroid|total)Magnitude(?P<magnitude>Apparent|Absolute):"+\
        "(?P<filter>[^:]+):(?P<frame>[^:]+)(?P<redshiftString>:z(?P<redshift>[\d\.]+))"+\
        "(?P<recent>:recent)?(?P<dust>:dust[^:]+)?(?P<system>:vega|:AB)?"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH
    

class GalacticusMagnitudes(object):

    def __init__(self,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise filters class
        self.FILTERS = GalacticusFilters()
        self.LUMINOSITIES = StellarLuminosities(galHDF5Obj)
        # Store Galacticus HDF5 object
        self.galHDF5Obj = galHDF5Obj
        return
    
    def createMagnitudeClass(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store dust optical depths information
        MAG = MagnitudeClass()
        MAG.datasetName = parseMagnitude(datasetName)
        # Identify HDF5 output
        MAG.outputName = self.galHDF5Obj.nearestOutputName(float(MAG.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(MAG.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            MAG.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(MAG.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            MAG.redshift = np.ones(ngals,dtype=float)*z
        return MAG

    def getVegaOffset(self,filterName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filterName not in self.FILTERS.vegaOffset.keys():
            self.FILTERS.load(filterName,path=filterFile)
        return self.FILTERS.vegaOffset[filterName]

    def setMagnitude(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create magnitude class
        MAG = self.createMagnitudeClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(MAG.datasetName.group('redshift')))
        # Check if luminosity already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(MAG.datasetName.group('redshift'))) and not overwrite:
            MAG.magnitude = np.array(HDF5OUT["nodeData/"+datasetName])
            return MAG
        # Create luminosity dataset name and extract luminosity
        luminosityDataset = MAG.datasetName.group('component')+"LuminositiesStellar:"+MAG.datasetName.group('filterName')+\
            ":"+MAG.datasetName.group('frame')+MAG.datasetName.group('redshiftString')
        if MAG.datasetName.group('recent') is not None:
            luminosityDataset = luminosityDataset + MAG.datasetName.group('recent')
        if MAG.datasetName.group('dust') is not None:
            luminosityDataset = luminosityDataset + MAG.datasetName.group('dust')
        MAG.luminosity = self.STELLAR.getLuminosity(luminosityDataset,z=z,overwrite=overwrite)
        # Compute absolute magnitude
        MAG.magnitude = -2.5*np.log10(MAG.luminosity+1.0e-40)
        # Convert to Vega magnitude if required (assume AB if nothing specified)
        if MAG.datasetName.group('system') is not None:
            if fnmatch.fnmatch(MAG.datasetName.group('system').lower(),"vega"):
                MAG.magnitude += self.getVegaOffset(MAG.datasetName.group('filterName'))
        # Convert to apparent magnitude if required
        if fnmatch.fnmatch(MAG.datasetName.group('magnitude').lower(),"app*"):
            distModulus = self.galHDF5Obj.cosmology.band_corrected_distance_modulus(MAG.redshift)
            MAG.magntiude += distModulus
        return MAG
        
    def getMagnitude(self,datasetName,z=None,overwrite=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        MAG = self.setMagnitude(datasetName,z=z,overwrite=overwrite)
        return MAG.magnitude

