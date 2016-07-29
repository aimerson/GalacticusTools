#! /usr/bin/env python

import sys,re
import fnmatch
import numpy as np
from .hdf5 import HDF5
from .io import GalacticusHDF5
from .Luminosities import Get_Luminosity
from .constants import speedOfLight,luminositySolar,luminosityAB,angstrom
from .utils.sorting import natural_sort_key
from .config import *
from .cosmology import Cosmology

##########################################################
# EMISSION LINES CLASS
##########################################################


class emissionLines(HDF5):
    
    def __init__(self):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initalise HDF5 class and open emissionLines.hdf5 file
        super(emissionLines, self).__init__(galacticusPath+"/data/hiiRegions/emissionLines.hdf5",'r')        
        # Extract names and properties of lines
        self.lines = list(map(str,self.fileObj["lines"].keys()))
        self.wavelengths = {}
        for l in self.lines:
            self.wavelengths[l] = self.readAttributes("lines/"+l,required=["wavelength"])["wavelength"]        
        return

    def getWavelength(self,line):
        index = self.lines.index(line)
        return self.wavelengths[index]



##########################################################
# MISC. FUNCTIONS
##########################################################

def getLineNames():    
    #lines = ["balmerAlpha6563","balmerBeta4861",\
    #             "oxygenII3726","oxygenII3729",\
    #             "oxygenIII4959","oxygenIII5007",\
    #             "nitrogenII6584",\
    #             "sulfurII6731","sulfurII6716"]
    lines = emissionLines().lines
    return lines

def availableLines(galHDF5Obj,z,frame=None,component=None,dust=None):    
    # Extract list of all emission line luminosities
    allLines = fnmatch.filter(galHDF5Obj.availableDatasets(z),"*LineLuminosity:*")
    # Select only for specified component?
    if component is not None:
        allLines = fnmatch.filter(allLines,component.lower()+"*")
    # Select only for specified frame?        
    if frame is not None:
        allLines = fnmatch.filter(allLines,"*:"+frame.lower()+":*")
    # Select only dust attenuated lines?
    if dust is not None:
        allLines = fnmatch.filter(allLines,"*:dustAtlas")
    # Extract emission line names
    avail = [ l.split(":")[1] for l in allLines ]
    return list(np.unique(avail))


def getLatexName(line):    
    name = "\mathrm{line}"
    if line.lower() == "balmeralpha6563":
        name = "\mathrm{H\\alpha}"
    elif line.lower() == "balmerbeta4861":
        name = "\mathrm{H\beta}"
    else:
        ones = "".join(fnmatch.filter(list(line),"I"))
        wave = line.split(ones)[-1]
        elem = line.split(ones)[0][0].upper()
        name = "\mathrm{"+elem+ones+"_{"+wave+"\\AA}}"
    return name



##########################################################
# EMISSION LINE LUMINOSITY
##########################################################

def Get_Total_Line_Luminosity(galHDF5Obj,z,datasetName,overwrite=False):
    funcname = sys._getframe().f_code.co_name
    if not datasetName.startswith("total"):
        print("WARNING! "+funcname+"(): '"+datasetName+"' is not a 'total' property!")
        return None
    else:
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if luminosity already calculated -- return if not wanting
        # to recalculate
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])
        else:
            # Extract disk and spheroid luminosities
            allprops = galHDF5Obj.availableDatasets(z)
            diskName = datasetName.replace("total","disk")
            diskLuminosity = Get_Line_Luminosity(galHDF5Obj,z,diskName)
            spheroidName = datasetName.replace("total","spheroid")
            spheroidLuminosity = Get_Line_Luminosity(galHDF5Obj,z,spheroidName)
            # Compute total luminosity and add to file
            totalLuminosity = diskLuminosity + spheroidLuminosity            
            galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,totalLuminosity)
            # Add appropriate attributes to new dataset
            attr = {"unitsInSI":luminositySolar}
            galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
            return totalLuminosity
        

def Get_Line_Luminosity(galHDF5Obj,z,datasetName,overwrite=False):
    funcname = sys._getframe().f_code.co_name
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if wanting a total luminosity -- if so pass to Get_Total_Luminosity
    if datasetName.startswith("total"):
        return Get_Total_Line_Luminosity(galHDF5Obj,z,datasetName,overwrite=overwrite)
    # Check if luminosity already calculated -- return if not wanting to recalculate
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        return np.array(out["nodeData/"+datasetName])
    else:
        # Calculate line luminosity -- need to finish this part!!!
        return None
    

##########################################################
# COMPUTE EQUIVALENT WIDTH
##########################################################

def Get_Equivalent_Width(galHDF5Obj,z,datasetName,overwrite=False):
    funcname = sys._getframe().f_code.co_name
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Check if already calculated -- return if not wanting to recalculate 
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite: 
        return np.array(out["nodeData/"+datasetName])

    # Extract information from dataset name
    lineName = datasetName.split(":")[0].split("_")[1]
    lineWavelength = re.sub("[^0-9]", "",lineName)
    resolution = datasetName.split(":")[0].split("_")[2]
    component = datasetName.split(":")[0].split("_")[0].replace("EmissionLineEW","")
    frame = datasetName.split(":")[1]
    redshift = datasetName.split(":")[2].replace("z","")

    # Locate names for three appropriate top-hat filters
    filterSearch = datasetName.replace("EmissionLineEW","LuminositiesStellar:emissionLineEW")
    if "_1band" in filterSearch:
       filterSearch = filterSearch.replace("_1band","")
    if "_2band" in filterSearch:
       filterSearch = filterSearch.replace("_2band","")       
    filterSearch = filterSearch.replace(lineName,lineName+"_*")
    allFilters = fnmatch.filter(galHDF5Obj.availableDatasets(z),filterSearch)
    if len(allFilters) < 3 and datasetName.startswith("total"):
        allFilters = fnmatch.filter(galHDF5Obj.availableDatasets(z),filterSearch.replace("total","disk"))
        for filter in allFilters:
            luminosity = Get_Luminosity(galHDF5Obj,z,filter.replace("disk","total"))
            del luminosity
        allFilters = fnmatch.filter(galHDF5Obj.availableDatasets(z),filterSearch)

    # Locate side band filters
    wavelengthCentral = [float(name.split(":")[1].split("_")[2]) for name in allFilters]
    mask = np.fabs(float(lineWavelength)-np.array(wavelengthCentral))>1.0
    centralBandName = list(np.array(allFilters)[np.invert(mask)])[0]
    sideBandNames = list(np.array(allFilters)[mask])
    sideBandNames.sort(key=natural_sort_key)
        
    # Store number of galaxies
    ngals = len(np.array(out["nodeData/nodeIndex"]))
    # Create array to store equivalent widths
    equivalentWidth = np.zeros(ngals)

    # Compute continuum luminosity according to desired method:    
    wavelengthMetres = float(lineWavelength)*angstrom
    if "_2band" in datasetName:
        lowWavelength = float(sideBandNames[0].split(":")[1].split("_")[2])*np.ones(ngals)
        uppWavelength = float(sideBandNames[1].split(":")[1].split("_")[2])*np.ones(ngals)
        lowContinuum = np.array(out["nodeData/"+sideBandNames[0]])
        uppContinuum = np.array(out["nodeData/"+sideBandNames[1]])
        continuum = (float(lineWavelength)-lowWavelength)/(uppWavelength-lowWavelength)
        continuum *= (uppContinuum-lowContinuum)
        continuum += lowContinuum
    elif "_1band" in datasetName:
        continuum = np.array(out["nodeData/"+centralBandName])        
    continuum *= luminosityAB*speedOfLight/(wavelengthMetres**2)
        
    # Compute emission line luminosity
    lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift
    lineLuminosity = Get_Line_Luminosity(galHDF5Obj,z,lineDatasetName)     
    lineLuminosity *= luminositySolar
    
    # Compute equivalent width
    nonZeroContinuum = continuum>0.0
    nonZeroEmissionLine = lineLuminosity>0.0
    nonZero = np.logical_and(nonZeroContinuum,nonZeroEmissionLine)
    continuum = continuum[nonZero]
    lineLuminosity = lineLuminosity[nonZero]
    width = (lineLuminosity/continuum)/angstrom
    np.place(equivalentWidth,nonZero,width)

    # Write equivalent width to file
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,equivalentWidth,overwrite=overwrite)
    attr = {"unitsInSI":angstrom}
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)

    return equivalentWidth

