#! /usr/bin/env python

import sys,os,re,fnamtch
from .io import GalacticusHDF5
from .Luminosities import Get_Luminosity
from .Filters import GalacticusFilters


class GalacticusMagnitudes(object):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initialise filters class
        self.FILTERS = GalacticusFilters()
        return

    def getMagnitude(self,galHDF5Obj,z,datasetName,overwrite=False,filterFile=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset name corresponds to an absolute magnitude
        MATCH = re.search(r"^magnitude([^:]+):([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:vega|:AB)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if magnitude already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])
        # Extract components
        component = MATCH.group(1)
        filter = MATCH.group(2)
        frame = MATCH.group(3)
        redshift = MATCH.group(4)
        dustExtension = MATCH.group(5)
        if dustExtension is None:
            dustExtension = ""
        vegaMagnitude = MATCH.group(6) == "vega"
        # Get luminosity dataset
        luminosityDataset = component.lower()+"LuminosityStellar:"+filter+":"+frame+"z"+z+dustExtension
        luminosity = Get_Luminosity(galHDF5Obj,z,luminosityDataset,overwrite=overwrite)
        # Compute magnitude
        magnitude = -2.5*np.log10(luminosity+1.0e-40)
        if vegaMagnitude:
            if filter not in self.FILTERS.vegaOffset.keys():                
                self.FILTERS.load(filter,path=filterFile)
            magnitude += self.FILTERS.vegaOffset[filter]
        # Add magnitude to file and return values 
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,magnitude)
        return magnitude
