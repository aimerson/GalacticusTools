#! /usr/bin/env python

import sys,re
import fnmatch
import numpy as np
import pkg_resources
from scipy.interpolate import interpn,interp1d
from scipy.integrate import romb
from .hdf5 import HDF5
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .Filters import GalacticusFilters
from .Luminosities import Get_Luminosity
from .constants import massSolar,luminositySolar,luminosityAB
from .constants import megaParsec,centi,Pi,erg,angstrom,speedOfLight
from .constants import massAtomic,atomicMassHydrogen,massFractionHydrogen
from .utils.sorting import natural_sort_key
from .config import *
from .cosmology import Cosmology
from .cloudy import cloudyTable


##########################################################
# EMISSION LINES CLASS
##########################################################

class GalacticusEmissionLines(object):
    
    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.CLOUDY = cloudyTable()
        self.FILTERS = GalacticusFilters()
        return

    def getLineNames(self):
        return self.CLOUDY.lines

    def getWavelength(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.CLOUDY.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        index = self.CLOUDY.lines.index(lineName)
        return self.CLOUDY.wavelengths[index]
                
    def getLuminosityMultiplier(self,datasetName,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        datasetInfo = datasetName.split(":")
        component = datasetInfo[0].replace("LineLuminosity","")
        lineName = datasetInfo[1]
        if datasetInfo[2]!="rest" and datasetInfo[2]!="observed":
            filterName = datasetInfo[2]
        else:
            filterName = None
        frame = fnmatch.filter(datasetInfo,"rest") + fnmatch.filter(datasetInfo,"observed")
        frame = frame[0]
        redshift = fnmatch.filter(datasetInfo,"z*")[0].replace("z","")
        # Return unity if no filter specified
        luminosityMultiplier = 1.0
        # Compute multiplier for filter
        if filterName is not None:
            FILTER = self.FILTERS.load(filterName,**kwargs).transmission
            lineWavelength = self.getWavelength(lineName)
            if frame == "observed":
                lineWavelength *= (1.0+float(redshift))
            luminosityMultiplier = 0.0
            lowLimit = FILTER.wavelength[FILTER.transmission>0].min()
            uppLimit = FILTER.wavelength[FILTER.transmission>0].max()                        
            if lineWavelength >= lowLimit and lineWavelength <= uppLimit:
                # Interpolate the transmission to the line wavelength  
                transmissionCurve = interp1d(FILTER.wavelength,FILTER.transmission)
                luminosityMultiplier = transmissionCurve(lineWavelength)                
                # Integrate a zero-magnitude AB source under the filter
                k = 10
                wavelengths = np.linspace(FILTER.wavelength[0],FILTER.wavelength[-1],2**k+1)
                deltaWavelength = wavelengths[1] - wavelengths[0]
                transmission = transmissionCurve(wavelengths)/wavelengths**2
                del transmissionCurve                
                transmission *= speedOfLight*luminosityAB/(angstrom*luminositySolar)
                filterLuminosityAB = romb(transmission,dx=deltaWavelength)
                # Compute the multiplicative factor to convert line
                # luminosity to luminosity in AB units in the filter
                luminosityMultiplier /= filterLuminosityAB
                # Galacticus defines observed-frame luminosities by
                # simply redshifting the galaxy spectrum without
                # changing the amplitude of F_nu (i.e. the compression
                # of the spectrum into a smaller range of frequencies
                # is not accounted for). For a line, we can understand
                # how this should affect the luminosity by considering
                # the line as a Gaussian with very narrow width (such
                # that the full extent of the line always lies in the
                # filter). In this case, when the line is redshifted
                # the width of the Gaussian (in frequency space) is
                # reduced, while the amplitude is unchanged (as, once
                # again, we are not taking into account the
                # compression of the spectrum into the smaller range
                # of frequencies). The integral over the line will
                # therefore be reduced by a factor of (1+z) - this
                # factor is included in the following line. Note that,
                # when converting this observed luminosity into an
                # observed flux a factor of (1+z) must be included to
                # account for compression of photon frequencies (just
                # as with continuum luminosities in Galacticus) which
                # will counteract the effects of the 1/(1+z) included
                # below.
                if frame == "observed":
                    luminosityMultiplier /= (1.0+float(redshift))
        return luminosityMultiplier
                
        
    def getLineLuminosity(self,galHDF5Obj,z,datasetName,overwrite=False,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check dataset name corresponds to a luminosity
        if not fnmatch.fnmatch(datasetName,"*LineLuminosity:*:z*"):
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])
        # Set mass (in solar masses) and lifetime (in Gyr) of HII regions
        massHIIRegion = 7.5e3 
        lifetimeHIIRegion = 1.0e-3 
        # Extract information from dataset name
        datasetInfo = datasetName.split(":")
        suffix = ":".join(datasetInfo[2:])
        component = datasetInfo[0].replace("LineLuminosity","")
        lineName = datasetInfo[1]
        frame = fnmatch.filter(datasetInfo,"rest") + fnmatch.filter(datasetInfo,"observed") 
        frame = frame[0]
        redshift = fnmatch.filter(datasetInfo,"z*")[0].replace("z","")
        # Compute various properties for interpolation over Cloudy tables
        gasMass = np.copy(out["nodeData/"+component+"MassGas"])
        radius = np.copy(out["nodeData/"+component+"Radius"])
        starFormationRate = np.copy(out["nodeData/"+component+"StarFormationRate"])
        abundanceGasMetals = np.copy(out["nodeData/"+component+"AbundancesGasMetals"])
        LyContinuum = np.copy(out["nodeData/"+component+"LuminositiesStellar:Lyc:"+suffix])
        HeContinuum = np.copy(out["nodeData/"+component+"LuminositiesStellar:HeliumContinuum:"+suffix])
        OxContinuum = np.copy(out["nodeData/"+component+"LuminositiesStellar:OxygenContinuum:"+suffix])
        # Useful masks to avoid dividing by zero etc.
        hasGas = gasMass > 0.0
        hasSize = radius > 0.0        
        hasFlux = LyContinuum > 0.0
        # i) compute metallicity
        metallicity = np.zeros_like(gasMass)
        np.place(metallicity,hasGas,np.log10(abundanceGasMetals[hasGas]/gasMass[hasGas]))
        # ii) compute hydrogen density
        densityHydrogen = np.zeros_like(gasMass)
        mask = np.logical_and(hasGas,hasSize)
        tmp = gasMass[mask]*massSolar/(radius[mask]*centi/megaParsec)**3
        tmp = np.log10(tmp/(4.0*Pi*massAtomic*atmicMassHydrogen*massFractionHydrogen))
        np.place(densityHydrogen,mask,np.copy(tmp))
        del tmp
        # iii) compute Lyman continuum luminosity
        ionizingFluxHydrogen = np.zeros_like(gasMass)
        tmp = np.log10(LyContinuum[hasFlux]) + 50.0
        np.place(ionizingFluxHydrogen,hasFlux,np.copy(tmp))
        del tmp
        # iv) compute luminosity ratios He/H and Ox/H
        ionizingFluxHeliumToHydrogen = np.zeros_lime(gasMass)
        tmp = np.log10(HeContinuum[hasFlux]/LyContinuum[hasFlux])
        np.place(ionizingFluxHeliumToHydrogen,hasFlux,np.copy(tmp))
        del tmp
        ionizingFluxOxygenToHydrogen = np.zeros_lime(gasMass)
        tmp = np.log10(OxContinuum[hasFlux]/LyContinuum[hasFlux])
        np.place(ionizingFluxOxygenToHydrogen,hasFlux,np.copy(tmp))
        del tmp
        # Check if returning raw line luminosity or luminosity under filter
        if filterName is not None:
            luminosityMultiplier = self.getLuminosityMultiplier(datasetName)
        # Find number of HII regions
        numberHIIRegion = starFormationRate*lifetimeHIIRegion/massHIIRegion
        # Convert the hydrogen ionizing luminosity to be per HII region
        ionizingFluxHydrogen -= np.log10(numberHIIRegion)
        # Interpolate over Cloudy tables to get luminosity per HII region
        lineLuminosity = self.CLOUDY.interpolate(lineName,metallicity,\
                                                     densityHydrogen,ionizingFluxHydrogen,\
                                                     ionizingFluxHeliumToHydrogen,\
                                                     ionizingFluxOxygenToHydrogen,**kwargs)
        # Convert to line luminosity in Solar luminosities (or AB maggies if a filter was specified)
        lineLuminosity *= luminosityMultiplier*numberHIIRegion*erg/luminositySolar
        # Add luminosity to file and return values
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,lineLuminosity)
        # Add appropriate attributes to new dataset
        attr = {"unitsInSI":luminositySolar}
        galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        return lineLuminosity

    def getTotalLineLuminosity(self,galHDF5Obj,z,datasetName,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if not datasetName.startswith("total"):
            print("WARNING! "+funcname+"(): '"+datasetName+"' is not a 'total' property!")
            return None
        # Check dataset name corresponds to a luminosity
        if not fnmatch.fnmatch(datasetName,"*LineLuminosity:*:z*"):
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if luminosity already calculated -- return if not
        # wanting to recalculate
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            return np.array(out["nodeData/"+datasetName])
        # Extract disk and spheroid luminosities
        allprops = galHDF5Obj.availableDatasets(z)
        diskName = datasetName.replace("total","disk")
        diskLuminosity = self.getLineLuminosity(galHDF5Obj,z,diskName)
        spheroidName = datasetName.replace("total","spheroid")
        spheroidLuminosity = self.getLineLuminosity(galHDF5Obj,z,spheroidName)
        # Compute total luminosity and add to file
        totalLuminosity = diskLuminosity + spheroidLuminosity
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,totalLuminosity)
        # Add appropriate attributes to new dataset
        attr = {"unitsInSI":luminositySolar}
        galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        return totalLuminosity





##########################################################
# MISC. FUNCTIONS
##########################################################

def getLineNames():    
    #lines = ["balmerAlpha6563","balmerBeta4861",\
    #             "oxygenII3726","oxygenII3729",\
    #             "oxygenIII4959","oxygenIII5007",\
    #             "nitrogenII6584",\
    #             "sulfurII6731","sulfurII6716"]
    lines = GalacticusEmissionLines().getLineNames()
    return lines

def computableLuminosities(availableDatasets):
    LyDisks = fnmatch.filter(availableDatasets,"diskLuminositiesStellar:Lyc*")
    computableDatasets = []
    components = "disk spheroid total".split()
    for dataset in LyDisks:
        haveContinua = fnmatch.filter(availableDatasets,dataset.replace("Lyc","HeliumContinuum"))>0 and \
            fnmatch.filter(availableDatasets,dataset.replace("Lyc","OxygenContinuum"))>0                        
        if haveContinua:
            suffix = ":".join(dataset.split(":")[2:])                
            dummy = [computableDatasets.append(comp+"LineLuminosity:"+line+":"+suffix) \
                         for comp in components for line in getLineNames()]
    return computableDatasets


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

