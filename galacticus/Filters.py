#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import romb
import pkg_resources
import xml.etree.ElementTree as ET
from .config import *
from .cloudy import cloudyTable
from .stellarPopulations import stellarPopulationSynthesisModel
from .GalacticusErrors import ParseError
from .xmlTree import formatFile


###############################################################################
# COMPUTE FILTER PROPERTIES
###############################################################################

def computeEffectiveWavelength(wavelength,transmission):
    return np.sum(wavelength*transmission)/np.sum(transmission)


###############################################################################
# AB/VEGA CONVERSIONS
###############################################################################


class VegaSpectrum(object):
    
    def __init__(self,spectrumFile=None):
        self.file = None
        self.spectrum = None
        self.description = None
        self.origin = None
        self.load(spectrumFile=spectrumFile)
        return

    def __call__(self):
        return self.spectrum
            
    def load(self,spectrumFile=None):
        # Find spectrum file
        if spectrumFile is None:
            default = "data/stellarAstrophysics/Vega/A0V_Castelli.xml"
            spectrumFile = pkg_resources.resource_filename(__name__,default)
        self.file = spectrumFile
        # Open file
        xmlStruct = ET.parse(spectrumFile)
        xmlRoot = xmlStruct.getroot()
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p} 
        data = xmlRoot.findall("datum")
        # Load spectrum
        self.spectrum = np.zeros(len(data),dtype=[("wavelength",float),("flux",float)])
        for i,datum in enumerate(data):
            self.spectrum["wavelength"][i] = float(datum.text.split()[0])
            self.spectrum["flux"][i] = float(datum.text.split()[1])
        self.spectrum = self.spectrum.view(np.recarray)
        isort = np.argsort(self.spectrum["wavelength"])
        self.spectrum["wavelength"] = self.spectrum["wavelength"][isort]
        self.spectrum["flux"] = self.spectrum["flux"][isort]
        # Load additional information
        self.description = xmlRoot.find("description").text
        self.origin = xmlRoot.find("origin").text
        return
        

class VegaOffset(object):
    
    def __init__(self,VbandFilterFile=None,kRomberg=8,**kwargs):
        super(VegaOffset,self).__init__()
        if VbandFilterFile is None:
            VbandFilterFile = pkg_resources.resource_filename(__name__,"data/filters/Buser_V.xml")
        self.load(VbandFilterFile,verbose=False,kRomberg=kRomberg,**kwargs)
        self.VEGA = VegaSpectrum()
        self.fluxVegaV = None
        self.fluxABV = None
        return
    
    def load(self,filterFile,verbose=False,kRomberg=8,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.fileVBand = filterFile        
        if verbose:
            print(classname+"(): Reading filter file '"+self.fileVBand+"'...")
        # Open xml file and load structure
        xmlStruct = ET.parse(self.fileVBand)
        xmlRoot = xmlStruct.getroot()
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}        
        # Read filter transmission
        response = xmlRoot.find("response")
        data = response.findall("datum")
        self.transmissionVBand = np.zeros(len(data),dtype=[("wavelength",float),("transmission",float)])
        for i,datum in enumerate(data):
            self.transmissionVBand["wavelength"][i] = float(datum.text.split()[0])
            self.transmissionVBand["transmission"][i] = float(datum.text.split()[1])        
        self.transmissionVBand = self.transmissionVBand.view(np.recarray)
        # Read header/information
        self.descriptionVBand = xmlRoot.find("description").text
        self.nameVBand = xmlRoot.find("name").text
        if "effectiveWavelength" in xmlMap.keys():
            self.effectiveWavelengthVBand = float(xmlRoot.find("effectiveWavelength").text)
        else:
            self.effectiveWavelengthVBand = computeEffectiveWavelength(self.transmission.wavelength,self.transmission.transmission)
        if "url" in xmlMap.keys():
            self.urlVBand = xmlRoot.find("url").text
        else:
            self.urlVBand = "unknown"
        if "origin" in xmlMap.keys():
            self.originVBand = xmlRoot.find("origin").text
        else:
            self.originVBand = "unknown"
        del xmlStruct, xmlRoot,xmlMap
        return

    def __call__(self,wavelength,transmission,kRomberg=8,**kwargs):
        return self.computeOffset(wavelength,transmission,kRomberg=kRomberg,**kwargs)
    
    def computeFluxes(self,wavelength,transmission,kRomberg=8,**kwargs):        
        # Interpolate spectrum and transmission data
        wavelengthJoint = np.linspace(wavelength.min(),wavelength.max(),2**kRomberg+1)
        deltaWavelength = wavelengthJoint[1] - wavelengthJoint[0]
        interpolateTransmission = interp1d(wavelength,transmission,**kwargs)
        interpolateFlux = interp1d(self.VEGA.spectrum.wavelength,self.VEGA.spectrum.flux,**kwargs)
        transmissionJoint = interpolateTransmission(wavelengthJoint)
        spectrumJoint = interpolateFlux(wavelengthJoint)
        # Get AB spectrum
        spectrumAB = 1.0/wavelengthJoint**2
        # Get the filtered spectrum
        filteredSpectrum = transmissionJoint*spectrumJoint
        filteredSpectrumAB = transmissionJoint*spectrumAB
        # Compute the integrated flux.        
        fluxVega = romb(filteredSpectrum,dx=deltaWavelength)
        fluxAB = romb(filteredSpectrumAB,dx=deltaWavelength)
        return fluxAB,fluxVega

    def computeOffset(self,wavelength,transmission,kRomberg=8,**kwargs):                
        # Compute fluxes for V-band magnitude if not already computed
        if self.fluxABV is None or self.fluxVegaV is None:
            self.fluxABV,self.fluxVegaV = self.computeFluxes(self.transmissionVBand.wavelength,\
                                                                 self.transmissionVBand.transmission,\
                                                                 kRomberg=kRomberg,**kwargs)
        # Compute fluxes for specified filter
        fluxAB,fluxVega = self.computeFluxes(wavelength,transmission,\
                                                 kRomberg=kRomberg,**kwargs)
        # Return Vega offset
        offset = 2.5*np.log10(fluxVega*self.fluxABV/self.fluxVegaV/fluxAB)
        return offset



###############################################################################
# FILTER FILES
###############################################################################

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
        
class FilterBaseClass(object):

    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return        
    
    def reset(self):
        self.file = None
        self.transmission = None
        self.description = None
        self.effectiveWavelength = None
        self.vegaOffset = None
        self.name = None
        self.origin = None
        self.url = None
        return

    def setEffectiveWavelength(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.transmission is None:
            raise ValueError(funcname+"(): no filter transmission has been set.")
        self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,\
                                                                  self.transmission.transmission)        
        return

    def setTransmission(self,wavelength,response):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store transmission
        if len(wavelength) != len(response):
            raise ValueError(funcname+"(): wavelength and response arrays are different length.")
        self.transmission = np.zeros(len(wavelength),dtype=[("wavelength",float),("transmission",float)]).view(np.recarray)        
        self.transmission.wavelength = wavelength
        self.transmission.transmission = response
        self.setEffectiveWavelength()
        return

    def setVegaOffset(self,vBandFile=None,kRomberg=8,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.transmission is None:
            raise ValueError(funcname+"(): no filter transmission has been set.")
        VEGA = VegaOffset(VbandFilterFile=vBandFile)
        self.vegaOffset = VEGA.computeOffset(self.transmission.wavelength,self.transmission.transmission,\
                                                 kRomberg=kRomberg,**kwargs)
        return



class Filter(FilterBaseClass):    
    
    def __init__(self):
        super(Filter,self).__init__()
        return        

    def __call__(self,filterFile,verbose=False,vBandFile=None,kRomberg=8,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.load(filterFile,verbose=verbose,vBandFile=vBandFile,kRomberg=kRomberg,**kwargs)
        return self

    def load(self,filterFile,verbose=False,vBandFile=None,kRomberg=8,**kwargs):
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
            self.setEffectiveWavelength()
        if "vegaOffset" in xmlMap.keys():
            self.vegaOffset = float(xmlRoot.find("vegaOffset").text)
        else:
            self.setVegaOffset(vBandFile=vBandFile,kRomberg=kRomberg,**kwargs)
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
            self.reportFilterInformation()
        return

    def reportFilterInformation(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
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

    def write(self,path,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create tree root
        root = ET.Element("filter")
        # Add name and other descriptions
        if self.name is None:
            raise ValueError(funcname+"(): must provide a name for the filter!")
        ET.SubElement(root,"name").text = self.name
        description = self.description
        if description is None:
            description = self.name
        ET.SubElement(root,"description").text = description
        origin = self.origin
        if origin is None:
            origin = "unknown"
        ET.SubElement(root,"origin").text = origin
        url = self.url
        if url is None:
            url = "unknown"
        ET.SubElement(root,"url").text = url
        # Add in response data
        if self.transmission is None:
            raise ValueError(funcname+"(): no transmission curve provided for filter!")
        RES = ET.SubElement(root,"response")
        for i in range(len(self.transmission.wavelength)):
            datum = "{0:7.3f} {1:9.7f}".format(self.transmission.wavelength[i],self.transmission.transmission[i])
            ET.SubElement(RES,"datum").text = datum
        # Compute effective wavlength and Vega offset if needed
        if self.effectiveWavelength is None:
            self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,\
                                                                      self.transmission.transmission)
        ET.SubElement(root,"effectiveWavelength").text = str(self.effectiveWavelength)
        if self.vegaOffset is None:
            vBandFilter = pkg_resources.resource_filename(__name__,"data/filters/Buser_V.xml")
            VO = VegaOffset(VbandFilterFile=vBandFilter)
            self.vegaOffset = VO.computeOffset(self.transmission.wavelength,self.transmission.transmission)
        ET.SubElement(root,"vegaOffset").text = str(self.vegaOffset)
        # Finalise tree and save to file
        tree = ET.ElementTree(root)
        if path is None:
            path = self.filtersDirectory
        path = path + "/"+self.name+".xml"
        if verbose:
            print(funcname+"(): writing filter to file: "+path)
        tree.write(path)
        formatFile(path)        
        return



###############################################################################
# TOP HAT FILTERS 
###############################################################################

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


class TopHat(FilterBaseClass):
    
    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(TopHat,self).__init__()
        self.wavelengthCentral = None
        self.wavelengthWidth = None
        return

    
    def __call__(self,filterName,vBandFile=None,kRomberg=8,transmissionSize=1000,edgesFraction=0.1,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if re.search("sedTopHat_(?P<center>[\d\.]+)_(?P<width>[\d\.]+)",filterName) is not None:
            self.setSEDTopHatFilter(filterName,vBandFile=vBandFile,kRomberg=kRomberg,transmissionSize=transmissionSize,\
                                        edgesFraction=edgesFraction,**kwargs)
        else:
            raise ValueError(funcname+"(): type of top hat filter not recognized!")
        return self
            

    def buildTransmissionCurve(self,centralWavelength,wavelengthWidth,transmissionSize=1000,\
                                   edgesFraction=0.1):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.wavelengthCentral = centralWavelength
        self.wavelengthWidth = wavelengthWidth
        fraction = 0.5 + edgesFraction
        lowerLimit = self.wavelengthCentral - self.wavelengthWidth*fraction
        upperLimit = self.wavelengthCentral + self.wavelengthWidth*fraction
        self.transmission = np.zeros(transmissionSize,dtype=[("wavelength",float),("transmission",float)])
        self.transmission = self.transmission.view(np.recarray)
        self.transmission.wavelength = np.linspace(lowerLimit,upperLimit,transmissionSize)
        lowerEdge = self.wavelengthCentral - self.wavelengthWidth/2.0
        upperEdge = self.wavelengthCentral + self.wavelengthWidth/2.0
        inside = np.logical_and(self.transmission.wavelength>=lowerEdge,self.transmission.wavelength<=upperEdge)
        np.place(self.transmission.transmission,inside,1.0)
        del inside
        self.setEffectiveWavelength()
        return
            
    def setSEDTopHatFilter(self,filterName,vBandFile=None,kRomberg=8,transmissionSize=1000,edgesFraction=0.1,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        
        searchString = "sedTopHat_(?P<center>[\d\.]+)_(?P<width>[\d\.]+)"
        MATCH = re.search(searchString,filterName)
        if MATCH is None:
            raise ParseError(funcname+"(): filter is not an SED top hat filter.")
        self.name = filterName
        self.buildTransmissionCurve(float(MATCH.group('center')),float(MATCH.group('width')),\
                                        transmissionSize=transmissionSize,edgesFraction=edgesFraction)
        self.origin = "Galacticus source code"
        self.description = "SED top hat filter centered on "+str(self.wavelengthCentral)+" Angstroms with width "+\
            str( self.wavelengthWidth)+" Angstroms."
        self.url = "None"
        self.setVegaOffset(vBandFile=vBandFile,kRomberg=8,**kwargs)
        return

        
class TopHat_v0(FilterBaseClass):
    
    def __init__(self,filterName,VegaObj=None,VbandFilterFile=None,kRomberg=8,\
                     transmissionArraySize=1000,verbose=False,**kwargs):
        super(TopHat,self).__init__()
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.name = filterName
        filterInfo = self.name.split("_")
        if fnmatch.fnmatch(self.name,"topHat*"):
            self.centralWavelength = float(filterInfo[1])
            self.resolution = float(filterInfo[2])
        elif fnmatch.fnmatch(self.name,"emissionLineContinuum*"):
            if len(filterInfo) == 4:
                self.centralWavelength = float(filterInfo[2])
                self.resolution = float(filterInfo[3])
            elif len(filterInfo) == 3:
                CLOUDY = cloudyTable()
                self.centralWavelength = CLOUDY.getWavelength(filterInfo[1])
                del CLOUDY
                self.resolution = float(filterInfo[2])
        else:
            raise ParseError(funcname+"(): filter '"+self.name+"' not a top hat filter!")
        # Compute transmission curve
        limits = getTopHatLimits(self.centralWavelength,self.resolution)
        offset = np.fabs(limits[1] - limits[0])*0.01
        self.transmission = np.zeros(transmissionArraySize,dtype=[("wavelength",float),("transmission",float)])
        self.transmission = self.transmission.view(np.recarray)
        self.transmission.wavelength = np.linspace(limits[0]-offset,limits[1]+offset,transmissionArraySize)
        inside = np.logical_and(self.transmission.wavelength>=limits[0],self.transmission.wavelength<=limits[1])
        np.place(self.transmission.transmission,inside,1.0)
        del inside
        # Compute effective wavelength
        self.effectiveWavelength = computeEffectiveWavelength(self.transmission.wavelength,self.transmission.transmission)
        # Compute AB-Vega offset
        if VegaObj is None:
            VegaObj = VegaOffset(VbandFilterFile=VbandFilterFile)
        self.vegaOffset = VegaObj.computeOffset(self.transmission.wavelength,self.transmission.transmission,\
                                                    kRomberg=kRomberg,**kwargs)
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


def buildSEDTopHatArray(lambdaMin,lambdaMax,lambdaWidth,redshift,SPS):
    # Create rest frame limits
    lambdaRestMin = lambdaMin/(1.0+redshift)
    lambdaRestMax = lambdaMax/(1.0+redshift)
    lambdaRestWidth = lambdaWidth/(1.0+redshift)
    # Manually build first filter
    lambdaCentral = lambdaRestMin
    lambdaWidth = np.maximum(lambdaRestWidth,SPS.wavelengthInterval(lambdaCentral))
    filterCentres = [lambdaCentral]
    filterWidths = [lambdaWidth]
    # Loop to create remaining filters
    while lambdaCentral < lambdaMax:
        lowerEdge = lambdaCentral + lambdaWidth/2.0
        tabulatedWidth = SPS.wavelengthInterval(lowerEdge)
        # Filter inside rest-frame range
        if lowerEdge < lambdaRestMax:
            lambdaWidth = lambdaRestWidth
            if tabulatedWidth > lambdaRestWidth:
                lambdaWidth = tabulatedWidth
            lambdaCentral = lowerEdge + lambdaWidth/2.0
        else:
            # Filter inside observer-frame range
            if lowerEdge > lambdaMin:
                lambdaWidth = lambdaWidth
                if tabulatedWidth > lambdaWidth:
                    lambdaWidth = tabulatedWidth
                lambdaCentral = lowerEdge + lambdaWidth/2.0
            else:
                # Filter is between rest-frame and observer frame
                lambdaCentral = lambdaMin
                lambdaWidth = lambdaWidth
                tabulatedWidth = SPS.wavelengthInterval(lambdaCentral)
                if tabulatedWidth > lambdaWidth:
                    lambdaWidth = tabulatedWidth
                if lambdaCentral - lambdaWidth/2.0 < lowerEdge:
                    tabulatedWidth = SPS.wavelengthInterval(lowerEdge)
                    lambdaWidth = lambdaWidth
                    if tabulatedWidth > lambdaWidth:
                        lambdaWidth = tabulatedWidth
                    lambdaCentral = lowerEdge + lambdaWidth/2.0
        filterCentres.append(lambdaCentral)
        filterWidths.append(lambdaWidth)
    return zip(filterCentres,filterWidths)
    



###############################################################################
# FILTERS CLASS
###############################################################################
        
class GalacticusFilters(object):
    
    def __init__(self,filtersDirectory=None):                
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filtersDirectory is None:
            filtersDirectory = pkg_resources.resource_filename(__name__,"/data/filters")+"/"
        self.filtersDirectory = filtersDirectory
        self.effectiveWavelengths = {}
        self.vegaOffset = {}
        self.filters = {}
        self.cloudyTableClass = cloudyTable()
        return

    def load(self,filterName,path=None,store=False,**kwargs):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if filterName not in self.filters.keys():        
            if fnmatch.fnmatch(filterName.lower(),"*tophat*") or fnmatch.fnmatch(filterName,"*emissionLine*"):
                T = TopHat()
                FILTER = T(filterName,**kwargs)                
            else:
                if path is None:
                    filterFile = filterName+".xml"                    
                    path = pkg_resources.resource_filename(__name__,"data/filters/"+filterFile)
                else:
                    path = path + "/" + filterName + ".xml"
                if not os.path.exists(path):
                    error = funcname+"(): Path to filter '"+filterName+"' does not exist!\n  Specified path = "+path
                    raise IOError(error)
                F = Filter()
                FILTER = F(path,**kwargs)
            self.effectiveWavelengths[filterName] = FILTER.effectiveWavelength
            self.vegaOffset[filterName] = FILTER.vegaOffset
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
            FILTER = self.load(filterName,path=path,store=store,verbose=verbose)
            del FILTER
            effectiveWavelength = self.effectiveWavelengths[filterName]
        if verbose:
            print(funcname+"(): Effective wavelength (rest frame) for filter '"+filterName + \
                      "' = {0:f} Angstroms".format(effectiveWavelength))
        return effectiveWavelength


    def create(self,name,response,description=None,effectiveWavelength=None,origin=None,path=None,\
                   url=None,vBandFilter=None,vegaOffset=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        F = Filter()
        F.name = name
        F.setTransmission(response["wavelength"],response["response"])
        if effectiveWavelength is not None:
            F.effectiveWavelength = effectiveWavelength
        F.origin = F.origin
        F.description = description
        F.url = url
        F.write(path=path,verbose=verbose)
        return

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
            if vBandFilter is None:
                vBandFilter = pkg_resources.resource_filename(__name__,"data/filters/Buser_V.xml")
            VO = VegaOffset(VbandFilterFile=vBandFilter)
            vegaOffset = VO.computeOffset(wavelength,transmission)
        ET.SubElement(root,"vegaOffset").text = str(vegaOffset)
        # Finalise tree and save to file
        tree = ET.ElementTree(root)
        if path is None:
            path = self.filtersDirectory
        path = path + "/"+name+".xml"
        if verbose:
            print(funcname+"(): writing filter to file: "+path)
        tree.write(path)
        formatFile(path)        
        return

