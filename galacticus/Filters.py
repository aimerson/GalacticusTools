#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from .config import *


def getTopHatLimits(wavelengthCentral,resolution,verbose=False):        
    funcname = sys._getframe().f_code.co_name
    if verbose:
        infoLine = "wavelengthCentral = {0:f}A  resolution = {1:f}A".format(wavelengthCentral,resolution)
        print(funcname+"(): "+infoLine)
    wavelengthRatio = (np.sqrt(4.0*resolution**2+1.0)+1.0)/(np.sqrt(4.0*resolution**2+1.0)-1.0)
    wavelengthMinimum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)-1.0)/2.0/resolution
    wavelengthMinimum /= wavelengthRatio
    wavelengthMaximum = wavelengthCentral*(np.sqrt(4.0*resolution**2+1.0)+1.0)/2.0/resolution
    wavelengthMaximum /= wavelengthRatio
    if verbose:
        infoLine = "wavelengthMinimum = {0:f}A  wavelengthMaximum = {1:f}A".format(wavelengthMinimum,wavelengthMaximum)
        print(funcname+"(): "+infoLine)
    return (wavelengthMinimum,wavelengthMaximum)



def computeEffectiveWavelength(wavelength,transmission):
    return np.sum(wavelength*transmission)/np.sum(transmission)

class Filter(object):    
    def __init__(self,filterFile,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = filterFile
        if verbose:
            print(classname+"(): Reading filter file '"+self.file+"'...")
        # Open xml file and load structure
        xmlStruct = ET.parse(self.file)
        xmlRoot = xmlStruct.getroot()
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}        
        # Read filter transmission
        response = xmlRoot.find("response")
        data = response.findall("datum")
        self.transmission = np.zeros(len(data),dtype=[("wavelength",float),("transmission",float)])
        for i,datum in enumerate(data):
            self.transmission["wavelength"][i] = float(datum.text.split()[0])
            self.transmission["transmission"][i] = float(datum.text.split()[1])        
        self.transmission = self.transmission.view(np.recarray)
        # Read header/information
        self.description = xmlRoot.find("description").text
        self.name = xmlRoot.find("name").text
        if "effectiveWavelength" in xmlMap.keys():
            self.effectiveWavelength = float(xmlRoot.find("effectiveWavelength").text)
        else:
            self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,self.transmission.transmission)
        self.vegaOffset = float(xmlRoot.find("vegaOffset").text)
        if "url" in xmlMap.keys():
            self.url = xmlRoot.find("url").text
        else:
            self.url = "unknown"
        if "origin" in xmlMap.keys():
            self.origin = xmlRoot.find("origin").text
        else:
            self.origin = "unknown"
        del xmlStruct, xmlRoot,xmlMap
        # Report filter information
        if verbose:
            print(classname+"(): Filter '"+self.name+"' attribrutes:")
            for k in self.__dict__.keys():
                if k is not "transmission":
                    print("        "+k+" = "+str(self.__dict__[k]))
            print(classname+"(): Filter '"+self.name+"' transmission curve:")
            print("                  Wavelength (A)                  Transmission")            
            infoLine = ["                   {0:f}                      {1:f}\n".format(w,t) \
                            for w,t in zip(self.transmission.wavelength,self.transmission.transmission)]
            print("".join(infoLine))
        return
            
        
class GalacticusFilters(object):
    
    def __init__(self,filtersDirectory=None):                
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filtersDirectory is None:
            self.filtersDirectory = galacticusPath+"/data/filters/"
        else:
            self.filtersDirectory = filtersDirectory
        self.effectiveWavelengths = {}
        self.filters = {}
        return

    def load(self,filterName,path=None,store=False,verbose=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filterName not in self.filters.keys():        
            if path is None:
                path = self.filtersDirectory+"/"+filterName+".xml"
            if not os.path.exists(path):
                error = funcname+"(): Path to filter '"+filterName+"' does not exist!\n  Specified path = "+path
                raise IOError(error)
            FILTER = Filter(path,verbose=verbose)
            self.effectiveWavelengths[filterName] = FILTER.effectiveWavelength
            if store:
                self.filters[filterName] = FILTER            
        else:
            FILTER = self.filters[filterName]
        return FILTER
    
    def getEffectiveWavelength(self,filterName,path=None,store=False,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filterName in self.effectiveWavelengths.keys():
            effectiveWavelength = self.effectiveWavelengths[filterName]
        else:
            if fnmatch.fnmatch(filterName,"*topHat*") or fnmatch.fnmatch(filterName,"*emissionLine*"):                            
                filterInfo = filterName.split("_")
                if fnmatch.fnmatch(filterName,"*topHat*"):
                    wavelength = filterInfo[1]
                    resolution = filterInfo[2]
                else:
                    wavelength = filterInfo[2]
                    resolution = filterInfo[3]
                if verbose:
                    infoLine = "filter={0:s}  wavelength={1:s}  resolution={2:s}".format(filter,wavelength,resolution)
                    print(funcname+"(): Top hat filter information:\n        "+infoLine)
                topHatLimits = getTopHatLimits(float(wavelength),float(resolution))
                effectiveWavelength = (topHatLimits[0]+topHatLimits[1])/2.0
            else:
                FILTER = self.load(filterName,path=path,store=store,verbose=verbose)
                del FILTER
                effectiveWavelength = self.effectiveWavelengths[filterName]
        if verbose:
            print(funcname+"(): Effective wavelength (rest frame) for filter '"+filterName + \
                      "' = {0:f} Angstroms".format(effectiveWavelength))
        return effectiveWavelength

