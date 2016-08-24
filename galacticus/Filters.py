#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romb
import pkg_resources
import xml.etree.ElementTree as ET
from .config import *
from .EmissionLines import cloudyTable
from .parameters import formatParametersFile

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



def getVegaSpectrum(specFile=None):
    if specFile is None:
        specFile = pkg_resources.resource_filename(__name__,"data/stellarAstrophysics/Vega/A0V_Castelli.xml")
        #specFile = galacticusPath+"/data/stellarAstrophysics/vega/A0V_Castelli.xml"
    # Load Vega spectrum
    xmlStruct = ET.parse(specFile)
    xmlRoot = xmlStruct.getroot()
    xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p} 
    data = xmlRoot.findall("datum")
    spectrum = np.zeros(len(data),dtype=[("wavelength",float),("flux",float)])
    for i,datum in enumerate(data):
        spectrum["wavelength"][i] = float(datum.text.split()[0])
        spectrum["flux"][i] = float(datum.text.split()[1])
    isort = np.argsort(spectrum["wavelength"])
    spectrum["wavelength"] = spectrum["wavelength"][isort]
    spectrum["flux"] = spectrum["flux"][isort]
    return spectrum.view(np.recarray)


def computeEffectiveWavelength(wavelength,transmission):
    return np.sum(wavelength*transmission)/np.sum(transmission)


def getFilterTransmission(filterFile):
    # Open xml file and load structure
    xmlStruct = ET.parse(filterFile)
    xmlRoot = xmlStruct.getroot()
    xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}
    # Read filter transmission
    response = xmlRoot.find("response")
    data = response.findall("datum")
    transmission = np.zeros(len(data),dtype=[("wavelength",float),("transmission",float)])
    for i,datum in enumerate(data):
        transmission["wavelength"][i] = float(datum.text.split()[0])
        transmission["transmission"][i] = float(datum.text.split()[1])
    return transmission.view(np.recarray)


class VegaOffset(object):
    
    def __init__(self,VbandFilterFile=None):
        if VbandFilterFile is None:
            VbandFilterFile = galacticusPath+"/data/filters/Buser_V.xml"
        self.transmissionV = getFilterTransmission(VbandFilterFile)
        self.vegaSpectrum = getVegaSpectrum()
        self.fluxVegaV = None
        self.fluxABV = None
        return

    def computeFluxes(self,wavelength,transmission,kRomberg=8,**kwargs):        
        # Interpolate spectrum and transmission data
        wavelengthJoint = np.linspace(wavelength.min(),wavelength.max(),2**kRomberg+1)
        deltaWavelength = wavelengthJoint[1] - wavelengthJoint[0]
        interpolateTransmission = interp1d(wavelength,transmission,**kwargs)
        interpolateFlux = interp1d(self.vegaSpectrum.wavelength,self.vegaSpectrum.flux,**kwargs)
        transmissionJoint = interpolateTransmission(wavelengthJoint)
        spectrumJoint = interpolateFlux(wavelengthJoint)
        # Get AB spectrum
        spectrumAB = 1.0/wavelengthJoint**2
        # Get the filtered spectrum
        filteredSpectrum = transmissionJoint*spectrumJoint;
        filteredSpectrumAB = transmissionJoint*spectrumAB;
        # Compute the integrated flux.        
        fluxVega = romb(filteredSpectrum,dx=deltaWavelength)
        fluxAB = romb(filteredSpectrumAB,dx=deltaWavelength)
        return (fluxAB,fluxVega)

    def computeOffset(self,wavelength,transmission,kRomberg=8,**kwargs):                
        # Compute fluxes for V-band magnitude if not already computed
        if self.fluxABV is None or self.fluxVegaV is None:
            wavelengthV = self.transmissionV.wavelength
            transmissionV = self.transmissionV.transmission
            (ABV,VegaV) = self.computeFluxes(wavelengthV,transmissionV,\
                                                 kRomberg=kRomberg,**kwargs)
            self.fluxABV = ABV
            self.fluxVegaV = VegaV
            del wavelengthV,transmissionV
        # Compute fluxes for specified filter
        (fluxAB,fluxVega) = self.computeFluxes(wavelength,transmission,\
                                                   kRomberg=kRomberg,**kwargs)
        # Return Vega offset
        return 2.5*np.log10(fluxVega*self.fluxABV/self.fluxVegaV/fluxAB)
    
        
class Filter(object):    
    def __init__(self,filterFile,verbose=False,VegaObj=None,VbandFilterFile=None,kRomberg=100,**kwargs):
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
        if "vegaOffset" in xmlMap.keys():
            self.vegaOffset = float(xmlRoot.find("vegaOffset").text)
        else:
            if VegaObj is None:
                VegaObj = VegaOffset(VbandFilterFile=VbandFilterFile)
            self.vegaOffset = VegaObj.computeOffset(self.transmission.wavelength,self.transmission.transmission,\
                                                        kRomberg=kRomberg,**kwargs)
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


def createFilter(filePath,name,response,description=None,origin=None,url=None,\
                     effectiveWavelength=None,vegaOffset=None,vBandFilter=None):    
    # Create tree root
    root = ET.Element("filter")
    # Add name and other descriptions
    ET.SubElement(root,"name").text = name
    if description is None:
        description = name
    ET.SubElement(root,"description").text = description
    if origin is None:
        origin = "unknown"
    ET.SubElement(root,"origin").text = origin
    if url is None:
        url = "unknown"
    ET.SubElement(root,"url").text = url
    # Add in response data
    RES = ET.SubElement(root,"response")
    dataSize = len(response["wavelength"])
    for i in range(dataSize):
        wavelength = response["wavelength"][i]
        transmission = response["response"][i]
        datum = "{0:7.3f} {1:9.7f}".format(wavelength,transmission)
        ET.SubElement(RES,"datum").text = datum
    # Compute effective wavlength and Vega offset if needed
    if effectiveWavelength is None:
        wavelength = response["wavelength"]
        transmission = response["response"]
        effectiveWavelength = computeEffectiveWavelength(wavelength,transmission)
    ET.SubElement(root,"effectiveWavelength").text = str(effectiveWavelength)
    if vegaOffset is None:
        wavelength = response["wavelength"]
        transmission = response["response"]
        VO = VegaOffset(VbandFilterFile=vBandFilter)
        vegaOffset = VO.computeOffset(wavelength,transmission)
    ET.SubElement(root,"vegaOffset").text = str(vegaOffset)
    # Finalise tree and save to file
    tree = ET.ElementTree(root)
    tree.write(filePath)
    formatParametersFile(filePath)
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
        self.cloudyTableClass = cloudyTable()
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
                    if fnmatch.fnmatch(filterName,"*emissionLineContinuumCentral*"):
                        wavelength = self.cloudyTableClass.getWavelength(filterInfo[1])
                        resolution = filterInfo[2]
                        pass
                    elif fnmatch.fnmatch(filterName,"*emissionLineContinuumOffset*"):
                        wavelength = filterInfo[3]
                        resolution = filterInfo[4]
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







