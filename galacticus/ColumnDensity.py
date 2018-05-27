#! /usr/bin/env python

import numpy as np
from scipy.special import digamma
from .io import GalacticusHDF5
from .Inclination import getInclination
from .galaxyProperties import DatasetClass
from .constants import Pi
from .constants import megaParsec,massHydrogen,massSolar,hydrogenByMassPrimordial



def parseColumnDensity(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)ColumnDensity"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class ColumnDensityClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,columnDensity=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(ColumnDensityClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.columnDensity = columnDensity
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.columnDensity = None
        return

    
class ColumnDensities(object):

    def __init__(self,galHDf5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDf5Obj = galHDf5Obj
        self.verbose = verbose
        # Evaluate conversion factor from mass column density to hydrogen column density.
        hecto = 1.00000000000e+02
        self.hydrogenFactor = hydrogenByMassPrimordial*massSolar/(massHydrogen*(megaParsec*hecto)**2)
        return
    
    def createColumnDensityClass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store column density information
        DENS = ColumnDensityClass()
        DENS.datasetName = parseColumnDensity(datasetName)
        # Identify HDF5 output
        DENS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            DENS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            DENS.redshift = np.ones(ngals,dtype=float)*redshift
        return DENS

    def getSpheroidColumnDensity(self,z,spheroidCutoff=0.1):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Extract gas mass and radius
        massGas = np.array(HDF5OUT["nodeData/spheroidMassGas"])
        radius = np.array(HDF5OUT["nodeData/spheroidRadius"])
        sigmaSpheroid = np.copy(np.zeros_like(radius))
        # Create mask to mask out galaxies with no bulge
        mask = radius > 0.0
        massGas = massGas[mask]
        radius = radius[mask]        
        # Compute gas density
        densityCentral = massGas/(2.0*Pi*radius**3) 
        # Compute column density
        radiusMinimum = spheroidCutoff*radius;
        radiusMinimumDimensionless = radiusMinimum/radius;        
        sigma = 0.5*np.copy(densityCentral)*np.copy(radius)
        sigma *= (-(3.0+2.0*radiusMinimumDimensionless)/\
            (1.0+radiusMinimumDimensionless)**2+\
            2.0*np.log(1.0+1.0/radiusMinimumDimensionless))
        np.place(sigmaSpheroid,mask,np.copy(sigma))
        # Tidy up and exit
        del massGas,radius,densityCentral
        del radiusMinimum,radiusMinimumDimensionless,sigma
        return sigmaSpheroid*self.hydrogenFactor

    def getDiskColumnDensity(self,z,diskHeightRatio=0.1):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Extract gas mass and radius
        massGas = np.array(HDF5OUT["nodeData/diskMassGas"])
        radius = np.array(HDF5OUT["nodeData/diskRadius"])
        sigmaSpheroid = np.copy(np.zeros_like(radius))
        # Create mask to mask out galaxies with no disk
        mask = radius > 0.0
        massGas = massGas[mask]
        radius = radius[mask]        
        densityCentral = massGas/(4.0*Pi*radius**3*diskHeightRatio)
        # Get inclination of disk
        inclination = getInclination(self.galHDF5Obj,z,degrees=False)[mask]
        inclinationHeight = np.fabs(np.tan(inclination))*diskHeightRatio
        # Compute column density to center of disk.
        digamma1 = digamma(-inclinationHeight/4.0)
        digamma2 = digamma(0.5-inclinationHeight/4.0)
        sigma = 0.5*densityCentral*radius*np.sqrt(1.0+1.0/inclination**2)
        sigma *= (inclinationHeight*(digamma1-digamma2)-2.0)
        np.place(sigmaDisk,mask,np.copy(sigma))
        # Tidy up
        del massGas,radius,densityCentral,inclination,inclinationHeight
        del digamma1,digamma2,sigma
        return sigmaDisk*self.hydrogenFactor


    def setColumnDensity(self,datasetName,z,overwrite=False,diskHeightRatio=0.1,spheroidCutoff=0.1):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create column density class
        DENS = createColumnDensityClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if  already available
        if datasetName in self.galHDF5Obj.availableDatasets(z) and not overwrite:
            DENS.columnDensity = np.array(HDF5OUT["nodeData/"+datasetName])
            return DENS
        # Compute column density
        if datasetName.startswith("total"):            
            kwargs = {}
            kwargs["overwrite"] = overwrite
            kwargs["diskHeightRatio"] = diskHeightRatio
            kwargs["spheroidCutoff"] = spheroidCutoff
            DENS.columnDensity = self.setColumnDensity(datasetName.replace("total","disk"),z,**kwargs) + \
                self.getColumnDensity(datasetName.replace("total","spheroid"),z,**kwargs)            
        else:
            if fnmatch.fnmatch(DENS.datasetName.group('component'),'disk'):
                DENS.columnDensity = self.getDiskColumnDensity(z,diskHeightRatio=diskHeightRatio)
            if fnmatch.fnmatch(DENS.datasetName.group('component'),'spheroid'):
                DENS.columnDensity = self.getSpheroidColumnDensity(z,spheroidCutoff=spheroidCutoff)
        return DENS

    def getColumnDensity(self,datasetName,z,overwrite=False,diskHeightRatio=0.1,spheroidCutoff=0.1):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DENS = self.setColumnDensity(datasetName,z,overwrite=overwrite,diskHeightRatio=diskHeightRatio,\
                                         spheroidCutoff=spheroidCutoff)
        return DENS.columnDensity



