#! /usr/bin/env python

import sys,re
from .agnSpectra import agnSpectralTables
from ..Luminosities import LuminosityClass
from ..constants import massSolar,speedOfLight,luminositySolar,gigaYear


def parseAGNLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract dataset name information
    searchString = "^agnLuminosity:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<absorption>:noAbsorption)?"+\
        "(?P<alphaString>:alpha(?P<alpha>[0-9\-\+\.]+))?$"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class AGNLuminosityClass(LuminosityClass):

    def __init__(self,datasetName=None,luminosity=None,\
                     redshift=None,outputName=None):
        super(AGNLuminosityClass,self).__init__(datasetName=datasetName,luminosity=luminosity,\
                                                    redshift=redshift,outputName=outputName)
        return


class AGNLuminosities(object):

    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        self.verbose = verbose
        self.SPECTRA = agnSpectralTables
        return


    def createAGNLuminosityClass(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store dust optical depths information
        AGN = AGNLuminosityClass()
        AGN.datasetName = parseAGNLuminosity(datasetName)
        # Identify HDF5 output
        AGN.outputName = self.galHDF5Obj.nearestOutputName(float(AGN.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(AGN.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            AGN.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(AGN.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            AGN.redshift = np.ones(ngals,dtype=float)*z
        return AGN


    def getBolometricLuminosity(self,z,zeroCorrection=1.0e-10):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        accretionRate = np.array(HDF5OUT["nodeData/blackHoleAccretionRate"])
        efficency = np.array(HDF5OUT['nodeData/blackHoleRadiativeEfficiency'])
        lumBol = efficency*accretionRate*massSolar/gigaYear*(speedOfLight**2)/luminositySolar        
        lumBol = np.maximum(lumBol,zeroCorrection)
        return np.log10(lumBol)


    def setLuminosity(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create AGN class
        AGN = self.createAGNLuminosityClass(datasetName)
        
    
