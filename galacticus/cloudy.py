#! /usr/bin/env python

import sys
import fnmatch
import numpy as np
import pkg_resources
from scipy.interpolate import interpn,interp1d
from scipy.integrate import romb
from .hdf5 import HDF5


##########################################################
# EMISSION LINES TABLE FROM CLOUDY
##########################################################

class cloudyTable(HDF5):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        hdf5File = pkg_resources.resource_filename(__name__,"data/stellarAstrophysics/hiiRegions/emissionLines.hdf5")
        # Initalise HDF5 class and open emissionLines.hdf5 file
        super(cloudyTable, self).__init__(hdf5File,'r')
        # Extract names and properties of lines
        self.lines = list(map(str,self.fileObj["lines"].keys()))
        self.wavelengths = {}
        self.luminosities = {}
        for l in self.lines:
            self.wavelengths[l] = self.readAttributes("lines/"+l,required=["wavelength"])["wavelength"]
            self.luminosities[l] = self.readDatasets('lines',required=[l])[l]
        # Store interpolants
        self.interpolantNames = ["metallicity","densityHydrogen","ionizingFluxHydrogen",\
                    "ionizingFluxHeliumToHydrogen","ionizingFluxOxygenToHydrogen"]
        self.interpolants = ()
        for name in self.interpolantNames:
            values = np.log10(self.readDatasets('/',required=[name])[name])
            self.interpolants = self.interpolants + (values,)
        return

    def getWavelength(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        index = self.lines.index(lineName)
        return self.wavelengths[index]

    def interpolate(self,lineName,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        tableLuminosities = self.luminosities[lineName]
        ngals = len(metallicity)
        galaxyData = zip(metallicity,densityHydrogen,ionizingFluxHydrogen,ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
        if "bounds_error" not in kwargs.keys():
            kwargs["bounds_error"] = False
        if "fill_value" not in kwargs.keys():
            kwargs["fill_value"] = None
        luminosities = interpn(self.interpolants,tableLuminosities,galaxyData,**kwargs)
        return luminosities
