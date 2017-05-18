
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

    def __init__(self,verbose=False):
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
        # Set verbosity
        self.verbose = verbose
        return

    def getInterpolantValues(self,interpolantName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if interpolantName not in self.interpolantNames:
            raise ValueError(funcname+"(): interpolant '"+interpolantName+"'not recognised! Options are: "+\
                                 ",".join(self.interpolantNames))
        return np.log10(self.readDatasets('/',required=[interpolantName])[interpolantName])        

    def getWavelength(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        return self.wavelengths[lineName]


    def reportLimits(self,metallicity,densityHydrogen,ionizingFluxHydrogen,\
                        ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("-"*40)
        print("CLOUDY Interpolation Report:")
        print("i )METALLICITY")
        print("Galacticus Data (min,max,median) = "+str(metallicity.min())+", "+str(metallicity.max())+", "+str(np.median(metallicity)))
        print("CLOUDY Range (min,max) = "+str(self.interpolants[0].min())+", "+str(self.interpolants[0].max()))
        print(" ")
        print("ii) HYDROGEN DENSITY")
        print("Galacticus Data (min,max,median) = "+str(densityHydrogen.min())+", "+str(densityHydrogen.max())+", "+str(np.median(densityHydrogen)))
        print("CLOUDY Range (min,max) = "+str(self.interpolants[1].min())+", "+str(self.interpolants[1].max()))
        print(" ")
        print("iii) IONIZING HYDROGEN FLUX")
        print("Galacticus Data (min,max,median) = "+str(ionizingFluxHydrogen.min())+", "+str(ionizingFluxHydrogen.max())+", "+str(np.median(ionizingFluxHydrogen)))
        print("CLOUDY Range (min,max) = "+str(self.interpolants[2].min())+", "+str(self.interpolants[2].max()))
        print(" ")
        print("iv) HELIUM/HYDROGEN RATIO")
        print("Galacticus Data (min,max,median) = "+str(ionizingFluxHeliumToHydrogen.min())+", "+str(ionizingFluxHeliumToHydrogen.max())+", "+str(np.median(ionizingFluxHeliumToHydrogen)))
        print("CLOUDY Range (min,max) = "+str(self.interpolants[3].min())+", "+str(self.interpolants[3].max()))
        print(" ")
        print("v) OXYGEN/HYDROGEN RATIO")
        print("Galacticus Data (min,max,median) = "+str(ionizingFluxOxygenToHydrogen.min())+", "+str(ionizingFluxOxygenToHydrogen.max())+", "+str(np.median(ionizingFluxOxygenToHydrogen)))
        print("CLOUDY Range (min,max) = "+str(self.interpolants[4].min())+", "+str(self.interpolants[4].max()))
        print("-"*40)
        return

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
        if self.verbose:
            self.reportLimits(metallicity,densityHydrogen,ionizingFluxHydrogen,\
                                  ionizingFluxHeliumToHydrogen,ionizingFluxOxygenToHydrogen)
        luminosities = interpn(self.interpolants,tableLuminosities,galaxyData,**kwargs)
        return luminosities
