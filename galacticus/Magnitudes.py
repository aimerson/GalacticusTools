#! /usr/bin/env python

import sys,os,re,fnamtch
from .utils.progress import Progress
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .Luminosities import getLuminosity
from .Filters import GalacticusFilters


class GalacticusMagnitudes(object):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise filters class
        self.FILTERS = GalacticusFilters()
        return

    def getAbsoluteMagnitude(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,filterFile=None,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset name corresponds to an absolute magnitude
        MATCH = re.search(r"^magnitude([^:]+):([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:vega|:AB)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if magnitude already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            if progressObj is not None:
                progressObj.increment()
                progressObj.print_status_line()
            if returnDataset:
                return np.array(out["nodeData/"+datasetName])
            else:
                return
        # Extract components
        component = MATCH.group(1)
        filterName = MATCH.group(2)
        frame = MATCH.group(3)
        redshift = MATCH.group(4)
        dustExtension = MATCH.group(5)
        if dustExtension is None:
            dustExtension = ""
        vegaMagnitude = MATCH.group(6) == "vega"
        # Get luminosity dataset
        luminosityDataset = component.lower()+"LuminosityStellar:"+filterName+":"+frame+"z"+z+dustExtension
        luminosity = getLuminosity(galHDF5Obj,z,luminosityDataset,overwrite=overwrite)
        # Compute magnitude
        magnitude = -2.5*np.log10(luminosity+1.0e-40)
        if vegaMagnitude:
            if filterName not in self.FILTERS.vegaOffset.keys():                
                self.FILTERS.load(filterName,path=filterFile)
            magnitude += self.FILTERS.vegaOffset[filterName]
        # Add magnitude to file and return values 
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,magnitude)
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return magnitude
        return
