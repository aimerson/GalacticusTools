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
from .IonizingContinuua import IonizingContinuua
from .Filters import GalacticusFilters
from .Luminosities import getLuminosity
from .constants import massSolar,luminositySolar,luminosityAB,metallicitySolar
from .constants import megaParsec,centi,Pi,erg,angstrom,speedOfLight
from .constants import massAtomic,atomicMassHydrogen,massFractionHydrogen
from .utils.sorting import natural_sort_key
from .utils.logicFunctions import OR
from .utils.misc import TemporaryClass
from .config import *
from .cosmology import Cosmology
from .cloudy import cloudyTable


##########################################################
# EMISSION LINES CLASS
##########################################################

class emissionLinesBase(object):
    
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        # Create classes for CLOUDY, ionization continuua and filter information
        self.CLOUDY = cloudyTable(verbose=verbose)
        self.FILTERS = GalacticusFilters()
        self.IONISATION = IonizingContinuua()
        # Store units in SI units
        self.unitsInSI = luminositySolar
        return

    def getLineNames(self):
        return self.CLOUDY.lines

    def getWavelength(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.CLOUDY.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        return float(self.CLOUDY.wavelengths[lineName])        

        
class GalacticusEmissionLine(emissionLineBase):

    def __init__(self,galHDF5Obj,massHIIRegion=7.5e3,lifetimeHIIRegion=1.0e-3,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusEmissionLine, self).__init__(verbose=verbose)
        # Set properties for HII regions
        self.massHIIRegion = massHIIRegion
        self.lifetimeHIIRegion = lifetimeHIIRegion
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        self.haveLightconeRedshifts = False
        # Initialise variables to store line information
        self.datasetName = None
        self.lineName = None
        self.lineRedshift = None
        self.lineLuminosity = None        
        return

    def updateHIIRegions(self,massHIIRegion=7.5e3,lifetimeHIIRegion=1.0e-3):
        self.massHIIRegion = massHIIRegion
        self.lifetimeHIIRegion = lifetimeHIIRegion
        return

    def resetHDF5Output(self):
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        self.haveLightconeRedshifts = False
        return

    def setHDF5Output(self,z):        
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        if self.galHDF5.datasetExists("lightconeRedshift",self.redshift):
            self.haveLightconeRedshifts = True
        return

    def resetLineInformation(self):
        self.datasetName = None
        self.lineName = None
        self.lineRedshift = None
        self.lineLuminosity = None        
        return
        
    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.datasetName = re.search("^(disk|spheroid|total)LineLuminosity:([^:]+)(:[^:]+)??(:[^:]+)??:z([\d\.]+)(:recent)?$",datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        self.lineName = self.datasetName.group(2)
        if self.lineName not in self.CLOUDY.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        self.lineRedshift = self.datasetName.group(5)        
        return 
                        
    def getLuminosityMultiplier(self,lineName,component,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        filterName = self.datasetName.group(3)
        if filterName is not None:
            filterName = filterName.replace(":","")
        frame = self.datasetName.group(4).replace(":","")
        redshift = self.lineRedshift
        # Return unity if no filter specified
        luminosityMultiplier = 1.0
        # Compute multiplier for filter
        if filterName is not None:
            FILTER = self.FILTERS.load(filterName,**kwargs).transmission
            lineWavelength = self.getWavelength(lineName)
            if frame == "observed":
                lineWavelength *= (1.0+self.lineRedshift)
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
                    luminosityMultiplier /= (1.0+self.lineRedshift)
        return luminosityMultiplier


    def computeHydrogenDensity(self,gasMass,radius):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        densityHydrogen = np.zeros_like(gasMass)
        hasGas = gasMass > 0.0
        hasSize = radius > 0.0
        mask = np.logical_and(hasGas,hasSize)
        tmp = gasMass[mask]*massSolar/(radius[mask]*megaParsec/centi)**3
        tmp *= massFractionHydrogen/(4.0*Pi*massAtomic*atomicMassHydrogen)
        tmp = np.log10(tmp)
        np.place(densityHydrogen,mask,np.copy(tmp))
        del mask,tmp
        return densityHydrogen

    def computeLymanContinuumLuminosity(self,LyContinuum):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        ionizingFluxHydrogen = np.zeros_like(LyContinuum)
        hasFlux = LyContinuum > 0.0
        np.place(ionizingFluxHydrogen,hasFlux,np.log10(LyContinuum[hasFlux])+50.0)
        return ionizingFluxHydrogen

    def computeHydrogenLuminosityRatio(self,LyContinuum,XContinuum):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        ionizingFluxXToHydrogen = np.zeros_like(LyContinuum)
        hasFlux = np.logical_and(LyContinuum>0.0,XContinuum>0.0)
        np.place(ionizingFluxXToHydrogen,hasFlux,np.log10(XContinuum[hasFlux]/LyContinuum[hasFlux]))
        return ionizingFluxXToHydrogen


    def calculateLineLuminosity(self,lineName,component,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Extract dataset components
        frame = self.datasetName.group(4).replace(":","")
        recent = self.datasetName.group(6)
        if recent is None:
            recent = ""        
        # Get appropriate redshift value
        if not self.datasetName.group(5):
            redshiftString = self.galHDF5.getRedshiftString(self.redshift)
        else:
            redshiftString = self.galHDF5.getRedshiftString(self.datasetName.group(5))
        # Extract various properties for interpolation over Cloudy tables
        gasMass = np.copy(self.hdf5Output["nodeData/"+component+"MassGas"])
        radius = np.copy(self.hdf5Output["nodeData/"+component+"Radius"])
        starFormationRate = np.copy(self.hdf5Output["nodeData/"+component+"StarFormationRate"])
        abundanceGasMetals = np.copy(self.hdf5Output["nodeData/"+component+"AbundancesGasMetals"])
        LyDatasetName = component+"LymanContinuumLuminosity:"+redshiftString
        LyContinuum = self.IONISATION.computeIonizingLuminosity(self.galHDF5Obj,self.redshift,LyDatasetName,postProcessingInformation=recent)
        HeDatasetName = component+"HeliumContinuumLuminosity:"+redshiftString
        HeContinuum = self.IONISATION.computeIonizingLuminosity(self.galHDF5Obj,self.redshift,HeDatasetName,postProcessingInformation=recent)
        OxDatasetName = component+"OxygenContinuumLuminosity:"+redshiftString
        OxContinuum = self.IONISATION.computeIonizingLuminosity(self.galHDF5Obj,self.redshift,OxDatasetName,postProcessingInformation=recent)
        # Useful masks to avoid dividing by zero etc.
        #hasGas = gasMass > 0.0
        hasGas = np.logical_and(gasMass>0.0,abundanceGasMetals>0.0)
        hasSize = radius > 0.0        
        hasFlux = LyContinuum > 0.0
        formingStars = starFormationRate > 0.0
        # i) compute metallicity
        metallicity = np.zeros_like(gasMass)
        np.place(metallicity,hasGas,np.log10(abundanceGasMetals[hasGas]/gasMass[hasGas]))
        metallicity -= np.log10(metallicitySolar)
        # ii) compute hydrogen density
        densityHydrogen = self.computeHydrogenDensity(gasMass,radius)
        # iii) compute Lyman continuum luminosity        
        ionizingFluxHydrogen = self.computeLymanContinuumLuminosity(LyContinuum)
        # iv) compute luminosity ratios He/H and Ox/H
        ionizingFluxHeliumToHydrogen = self.computeHydrogenLuminosityRatio(LyContinuum,HeContinuum)
        ionizingFluxOxygenToHydrogen = self.computeHydrogenLuminosityRatio(LyContinuum,OxContinuum)
        # Check if returning raw line luminosity or luminosity under filter
        luminosityMultiplier = self.getLuminosityMultiplier(**kwargs)
        # Find number of HII regions        
        starFormationRate = np.maximum(starFormationRate,1.0e-20)
        numberHIIRegion = starFormationRate*self.lifetimeHIIRegion/self.massHIIRegion
        # Convert the hydrogen ionizing luminosity to be per HII region
        ionizingFluxHydrogen -= np.log10(numberHIIRegion)
        # Interpolate over Cloudy tables to get luminosity per HII region
        lineLuminosity = self.CLOUDY.interpolate(lineName,metallicity,\
                                                 densityHydrogen,ionizingFluxHydrogen,\
                                                 ionizingFluxHeliumToHydrogen,\
                                                 ionizingFluxOxygenToHydrogen,**kwargs)
        # Convert to line luminosity in Solar luminosities (or AB maggies if a filter was specified)        
        lineLuminosity *= luminosityMultiplier*numberHIIRegion*erg/luminositySolar
        lineLuminosity = np.maximum(lineLuminosity,0.0)
        return 
                
    def setLineInformation(self,datasetName,overwrite=False,z=None,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Reset line information
        self.resetLineInformation()        
        # Set datasetName
        self.setDatasetName(datasetName)                        
        # Check HDF5 snapshot specified        
        if z is not None:
            self.setHDF5Output(z)
        else:
            if self.hdf5Output is None:
                z = self.datasetName.group(5)
                if z is None:
                    errMsg = funcname+"(): no HDF5 output specified. Either specify the redshift "+\
                             "of the output or include the redshift in the dataset name."
                    raise RunTimeError(errMsg)
        # Set line redshift
        if self.haveLightconeRedshifts:
            self.lineRedshift = np.array(self.hdf5Output["nodeData/lightconeRedshift"])
        else:
            self.lineRedshift = float(self.redshift)
        # Check if luminosity already calculated
        if self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) and not overwrite:
            self.lineLuminosity = np.array(out["nodeData/"+datasetName])
            return        
        # Check if computing a total line luminosity or a disk/spheroid luminosity
        if fnmatch.fnmatch(self.datasetName.group(1),"total"):
            # Compute disk line luminosity
            self.lineLuminosity = self.calculateLineLuminosity(self.lineName,"disk",**kwargs) +\
                                  self.calculateLineLuminosity(self.lineName,"spheroid",**kwargs)
        else:
            # Compute line luminosity
            self.lineLuminosity = self.calculateLineLuminosity(self.lineName,self.datasetName.group(1),**kwargs)
        return
            
    def getLineLuminosity(self,datasetName,overwrite=False,z=None,progressObj=None,\
                          **kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.setLineInformation(datasetName,overwrite=overwrite,z=z,**kwargs)
        return self.lineLuminosity

        
    def writeLineLuminosityToFile(self,overwrite=False):
        if not self.datasetName.group(0) in self.galacticusOBJ.availableDatasets(self.redshift) or overwrite:
            out = self.galacticusOBJ.selectOutput(self.redshift)
            # Add luminosity to file
            self.galacticusOBJ.addDataset(out.name+"/nodeData/",self.datasetName.group(0),np.copy(self.lineLuminosity))
            # Add appropriate attributes to new dataset
            attr = {"unitsInSI":luminositySolar}
            self.galacticusOBJ.addAttributes(out.name+"/nodeData/"+self.datasetName.group(0),attr)
        return



class ContaminateEmissionLine(object):

    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = False
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.datasetName = None
        self.lineName = None
        self.lineLuminosity = None
        return

    def resetHDF5Output(self,galHDF5Obj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if galHDF5Obj:
            self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        return

    def setHDF5Output(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        return

    def resetLineInformation(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.lineName = None
        self.lineLuminosity = None
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.resetLineInformation()
        self.datasetName = re.search("^(disk|spheroid|total)LineLuminosity:([^:]+)(:[^:]+)??(:[^:]+)??:z([\d\.]+)(:recent)?(:dust[^:]+)?$",datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        self.lineName = self.datasetName.group(2)
        if self.lineName not in self.CLOUDY.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        return

    def extractLineLuminosity(self,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        datasetName = self.datasetName.replace(self.lineName,lineName)
        if not self.galHDF5.datasetExists(datasetName,self.redshift):
            raise RuntimeError(funcname+"(): "+datasetName+" not found in file.")        
        return np.array(self.hdf5Output["nodeData/"+datasetName])
    
    def addContaminant(self,lineName):
        self.lineLuminosity += self.extractLineLuminosity(lineName)
        return

    def setLineLuminosity(self,datasetName,z=None,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Remove list of contaminants
        contaminants = " ".join(fnmatch.filter(datasetName.split(":"),"contam_*"))
        contaminants = contaminants.replace("contam_","").split()
        contaminantFreeName = [name if not fnmatch.fnmatch(name,"contam_*") for name in datasetName.split(":")]
        contaminantFreeName = ":".join(contaminantFreeName)        
        self.setDatasetName(contaminantFreeName)
        # Check HDF5 snapshot specified
        if z is not None:
            self.setHDF5Output(z)
        else:
            if self.hdf5Output is None:
                z = self.datasetName.group(5)
                if z is None:
                    errMsg = funcname+"(): no HDF5 output specified. Either specify the redshift "+\
                             "of the output or include the redshift in the dataset name."
                    raise RunTimeError(errMsg)
                self.setHDF5Output(z)
        # Check if luminosity already calculated
        if self.galHDF5.datasetExists(datasetName,self.redshift) and not overwrite:
            self.lineLuminosity = np.array(self.hdf5Output["nodeData/"+datasetName])
            return
        # Set luminosity of uncontaminated line
        self.lineLuminosity = self.extractLineLuminosity(self.lineName)
        # Process contaminants
        dummy = [self.addContaminant(contam) for contam in contaminants]
        return

    def getLineLuminosity(self,datasetName,z=None,overwrite=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.setLineLuminosity(datasetName,z=z,overwrite=overwrite)
        return self.lineLuminosity



class GalacticusEmissionLines(object):
    
    def __init__(self,massHIIRegion=7.5e3,lifetimeHIIRegion=1.0e-3,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Create classes for CLOUDY, ionization continuua and filter information
        self.CLOUDY = cloudyTable(verbose=verbose)
        self.FILTERS = GalacticusFilters()
        self.IONISATION = IonizingContinuua()
        # Set properties for HII regions
        self.massHIIRegion = massHIIRegion
        self.lifetimeHIIRegion = lifetimeHIIRegion
        return

    def getLineNames(self):
        return self.CLOUDY.lines

    def getWavelength(self,lineName,redshift=0.0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lineName not in self.CLOUDY.lines:
            raise IndexError(funcname+"(): Line '"+lineName+"' not found!")
        return float(self.CLOUDY.wavelengths[lineName])*(1.0+redshift)
                
    #####################################################################################
    # LINE LUMINOSITIES
    #####################################################################################
    
    def parseLuminosityDataset(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        MATCH = re.search("^(disk|spheroid)LineLuminosity:([^:]+)(:[^:]+)??(:[^:]+)??:z([\d\.]+)(:recent)?(:contam_[^:]+)?$",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        return MATCH


    def getLuminosityMultiplier(self,datasetName,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract information from dataset name
        MATCH = self.parseLuminosityDataset(datasetName)
        component = MATCH.group(1).lower()
        lineName = MATCH.group(2)
        filterName = MATCH.group(3)
        if filterName is not None:
            filterName = filterName.replace(":","")
        frame = MATCH.group(4).replace(":","")
        redshift = MATCH.group(5)
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


    def computeHydrogenDensity(self,gasMass,radius):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        densityHydrogen = np.zeros_like(gasMass)
        hasGas = gasMass > 0.0
        hasSize = radius > 0.0
        mask = np.logical_and(hasGas,hasSize)
        tmp = gasMass[mask]*massSolar/(radius[mask]*megaParsec/centi)**3
        tmp *= massFractionHydrogen/(4.0*Pi*massAtomic*atomicMassHydrogen)
        tmp = np.log10(tmp)
        np.place(densityHydrogen,mask,np.copy(tmp))
        del mask,tmp
        return densityHydrogen

    def computeLymanContinuumLuminosity(self,LyContinuum):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        ionizingFluxHydrogen = np.zeros_like(LyContinuum)
        hasFlux = LyContinuum > 0.0
        np.place(ionizingFluxHydrogen,hasFlux,np.log10(LyContinuum[hasFlux])+50.0)
        return ionizingFluxHydrogen

    def computeHydrogenLuminosityRatio(self,LyContinuum,XContinuum):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        ionizingFluxXToHydrogen = np.zeros_like(LyContinuum)
        hasFlux = np.logical_and(LyContinuum>0.0,XContinuum>0.0)
        np.place(ionizingFluxXToHydrogen,hasFlux,np.log10(XContinuum[hasFlux]/LyContinuum[hasFlux]))
        return ionizingFluxXToHydrogen


    def calculateLineLuminosity(self,galHDF5Obj,z,datasetName,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check dataset name corresponds to a luminosity
        MATCH = self.parseLuminosityDataset(datasetName)
        # Extract dataset components
        component = MATCH.group(1).lower()
        lineName = MATCH.group(2)
        filterName = MATCH.group(3)
        if filterName is not None:
            filterName = filterName.replace(":","")
        frame = MATCH.group(4).replace(":","")
        redshift = MATCH.group(5)
        recent = MATCH.group(6)
        if recent is None:
            recent = ""        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Extract various properties for interpolation over Cloudy tables
        gasMass = np.copy(out["nodeData/"+component+"MassGas"])
        radius = np.copy(out["nodeData/"+component+"Radius"])
        starFormationRate = np.copy(out["nodeData/"+component+"StarFormationRate"])
        abundanceGasMetals = np.copy(out["nodeData/"+component+"AbundancesGasMetals"])
        LyDatasetName = component+"LymanContinuumLuminosity:z"+redshift
        LyContinuum = self.IONISATION.computeIonizingLuminosity(galHDF5Obj,z,LyDatasetName,postProcessingInformation=recent)
        HeDatasetName = component+"HeliumContinuumLuminosity:z"+redshift
        HeContinuum = self.IONISATION.computeIonizingLuminosity(galHDF5Obj,z,HeDatasetName,postProcessingInformation=recent)
        OxDatasetName = component+"OxygenContinuumLuminosity:z"+redshift
        OxContinuum = self.IONISATION.computeIonizingLuminosity(galHDF5Obj,z,OxDatasetName,postProcessingInformation=recent)
        # Useful masks to avoid dividing by zero etc.
        #hasGas = gasMass > 0.0
        hasGas = np.logical_and(gasMass>0.0,abundanceGasMetals>0.0)
        hasSize = radius > 0.0        
        hasFlux = LyContinuum > 0.0
        formingStars = starFormationRate > 0.0
        # i) compute metallicity
        metallicity = np.zeros_like(gasMass)
        np.place(metallicity,hasGas,np.log10(abundanceGasMetals[hasGas]/gasMass[hasGas]))
        metallicity -= np.log10(metallicitySolar)
        # ii) compute hydrogen density
        densityHydrogen = self.computeHydrogenDensity(gasMass,radius)
        # iii) compute Lyman continuum luminosity        
        ionizingFluxHydrogen = self.computeLymanContinuumLuminosity(LyContinuum)
        # iv) compute luminosity ratios He/H and Ox/H
        ionizingFluxHeliumToHydrogen = self.computeHydrogenLuminosityRatio(LyContinuum,HeContinuum)
        ionizingFluxOxygenToHydrogen = self.computeHydrogenLuminosityRatio(LyContinuum,OxContinuum)
        # Check if returning raw line luminosity or luminosity under filter
        if filterName is not None:
            luminosityMultiplier = self.getLuminosityMultiplier(datasetName)
        else:
            luminosityMultiplier = 1.0
        # Find number of HII regions        
        starFormationRate = np.maximum(starFormationRate,1.0e-20)
        numberHIIRegion = starFormationRate*self.lifetimeHIIRegion/self.massHIIRegion
        # Convert the hydrogen ionizing luminosity to be per HII region
        ionizingFluxHydrogen -= np.log10(numberHIIRegion)
        # Interpolate over Cloudy tables to get luminosity per HII region
        lineLuminosity = self.CLOUDY.interpolate(lineName,metallicity,\
                                                     densityHydrogen,ionizingFluxHydrogen,\
                                                     ionizingFluxHeliumToHydrogen,\
                                                     ionizingFluxOxygenToHydrogen,**kwargs)
        # Convert to line luminosity in Solar luminosities (or AB maggies if a filter was specified)        
        lineLuminosity *= luminosityMultiplier*numberHIIRegion*erg/luminositySolar
        lineLuminosity = np.maximum(lineLuminosity,0.0)
        return lineLuminosity


    def getLineLuminosity(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None,\
                              **kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Check if luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            if progressObj is not None:
                progressObj.increment()
                progressObj.print_status_line()
            if returnDataset:
                return np.array(out["nodeData/"+datasetName])
            else:
                return
        # Check if requesting pure or contaminated luminosity
        if fnmatch.fnmatch(datasetName,"*:contam_*"):
            contaminants = fnmatch.filter(datasetName.split(":"),"contam_*")[0]
            pureDatasetName = datasetName.replace(":"+contaminants,"")
        else:
            pureDatasetName = datasetName
            contaminants = None
        # Check whether computing disk/spheroid/total luminosity
        if pureDatasetName.startswith("disk") or pureDatasetName.startswith("spheroid"):
            lineLuminosity = self.calculateLineLuminosity(galHDF5Obj,z,pureDatasetName,**kwargs)
        else:
            # Compute total luminosity by summing disk and spheroid luminosities
            diskName = pureDatasetName.replace("totalLineLuminosity","diskLineLuminosity")
            spheroidName = pureDatasetName.replace("totalLineLuminosity","spheroidLineLuminosity")
            lineLuminosity = np.copy(self.getLineLuminosity(galHDF5Obj,z,diskName,overwrite=False,returnDataset=True,**kwargs)+\
                                     self.getLineLuminosity(galHDF5Obj,z,spheroidName,overwrite=False,returnDataset=True,**kwargs))
        # Apply contamination if requested            
        if contaminants is not None:
            contaminants = contaminants.replace("contam_","").split("_")
            lineName = pureDatasetName.split(":")[1]
            TEMP = TemporaryClass(lineLuminosity)
            dummy = [TEMP.updateArray(self.getLineLuminosity(galHDF5Obj,z,pureDatasetName.replace(lineName,contam),\
                                                                 overwrite=False,returnDataset=True,progressObj=None,**kwargs),np.add)\
                         for contam in contaminants]
            del dummy
            lineLuminosity = np.copy(TEMP.array)
            del TEMP
        # Add luminosity to file
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,np.copy(lineLuminosity))
        # Add appropriate attributes to new dataset
        attr = {"unitsInSI":luminositySolar}
        galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        # Print progress if requested
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        # Optionally return dataset array
        if returnDataset:
            return lineLuminosity
        del lineLuminosity
        return

    #####################################################################################
    # EQUIVALENT WIDTHS
    #####################################################################################

    
    def getTopHatWavelength(self,datasetName,lineWavelength,frame,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name                
        if "emissionLineContinuumCentral" in datasetName:
            wavelength = lineWavelength
        else:
            MATCH = re.search("^(disk|spheroid|total)LuminositiesStellar:emissionLineContinuumOffset_([^_]+)_([\d\.]+)_([\d\.]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:[^:]+)?$",datasetName)
            wavelength = float(MATCH.group(3))
            if frame == "observed":
                wavelength *= (1.0+float(z))
        return wavelength


    def computeContinuumLuminosity(self,galHDF5Obj,z,frame,searchPattern,lineName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Select redshift output
        out = galHDF5Obj.selectOutput(z)
        z = galHDF5Obj.nearestRedshift(z)
        # Get emission line wavelength        
        lineWavelength = self.getWavelength(lineName)
        if frame == "observed":
            lineWavelength *= (1.0+float(z))
        # Extract filters
        if searchPattern.startswith("total"):
            if len(fnmatch.filter(galHDF5Obj.availableDatasets(z),searchPattern))==0:
                diskPattern = searchPattern.replace("total","disk")
                topHatDatasetNames = fnmatch.filter(galHDF5Obj.availableDatasets(z),diskPattern)
                dummy = [getLuminosity(galHDF5Obj,z,datasetName.replace("disk","total"),overwrite=False,returnDataset=False,progressObj=None)\
                             for datasetName in topHatDatasetNames]
                del dummy
        topHatDatasetNames = fnmatch.filter(galHDF5Obj.availableDatasets(z),searchPattern)        
        # Extract wavelengths
        topHatWavelengths = np.array([float(self.getTopHatWavelength(topHatName,lineWavelength,frame,z)) for topHatName in topHatDatasetNames])
        if len(topHatDatasetNames) == 2:
            # Working with pair of top hat filters either side of emission line            
            iLow = np.argmin(topHatWavelengths)
            iUpp = np.argmax(topHatWavelengths)
            lowContinuum = np.array(out["nodeData/"+topHatDatasetNames[iLow]])
            uppContinuum = np.array(out["nodeData/"+topHatDatasetNames[iUpp]])
            diffContinuum = np.copy(uppContinuum) - np.copy(lowContinuum)
            wavelengthRatio = (float(lineWavelength)-topHatWavelengths[iLow])/(topHatWavelengths[iUpp]-topHatWavelengths[iLow])
            luminosity = diffContinuum*wavelengthRatio
            luminosity += np.copy(lowContinuum)
            del lowContinuum,uppContinuum
        elif len(topHatDatasetNames) == 1:
            luminosity = np.array(out["nodeData/"+topHatDatasetNames[0]])
        else:
            raise IndexError(funcname+"(): Unable to locate any top hat filters for computing continuum luminosity!")
        wavelengthMetres = float(lineWavelength)*angstrom
        luminosity *= luminosityAB*speedOfLight/(wavelengthMetres**2)
        return luminosity



    def getEquivalentWidth(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check property corresponds to equivalent width        
        MATCH = re.search("^(disk|spheroid|total)EquivalentWidth:([^:]+):([^:]+)_([\d\.]+):([^:]+):z([\d\.]+)(:contam_[^:]+)?(:dust[^:]+)?$",datasetName)        
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract dataset information
        component = MATCH.group(1).lower()
        lineName = MATCH.group(2)
        calculationMethod = MATCH.group(3)
        resolution = MATCH.group(4)
        frame = MATCH.group(5)
        redshift = MATCH.group(6)
        dust = MATCH.group(7)
        if dust is None:
            dust = ""
        contam = MATCH.group(8)
        if contam is None:
            contam = ""            
        # Get emission line wavelength
        lineWavelength = self.getWavelength(lineName)
        if frame == "observed":
            lineWavelength *= (1.0+float(redshift))
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)                        
        # Extract emission line luminosity
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+contam+dust
        if lineDatasetName.startswith("total"):
            self.getLineLuminosity(galHDF5Obj,z,lineDatasetName,overwrite=False,returnDataset=False)
        if lineDatasetName not in out["nodeData"].keys():
            raise KeyError(funcname+"(): emission line luminosity '"+lineDatasetName+"' cannot be found!") 
        lineLuminosity = self.getLineLuminosity(galHDF5Obj,z,lineDatasetName,overwrite=False,returnDataset=True)
        lineLuminosity *= luminositySolar
        # Compute continuum luminosity
        if calculationMethod.lower() in ["pair","offset"]:
            searchPattern = component+"LuminositiesStellar:emissionLineContinuumOffset_"+lineName+"_*_"+\
                resolution+":"+frame+":z"+redshift+dust
        elif calculationMethod.lower() in ["central"]:
            searchPattern = component+"LuminositiesStellar:emissionLineContinuumCentral_"+lineName+"_"+\
                resolution+":"+frame+":z"+redshift+dust
        else:
            raise ValueError(funcname+"(): calculation method not recognised! Should be 'Central' or 'Offset'.")
        continuumLuminosity = self.computeContinuumLuminosity(galHDF5Obj,float(redshift),frame,searchPattern,lineName)
        # Compute equivalent width        
        nonZeroContinuum = continuumLuminosity>0.0    
        equivalentWidth = np.ones_like(lineLuminosity)*-999.9
        mask = np.logical_and(nonZeroContinuum,lineLuminosity>=0.0)
        np.place(equivalentWidth,mask,(lineLuminosity[mask]/continuumLuminosity[mask]))
        equivalentWidth /= angstrom
        # Write equivalent width to file
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,equivalentWidth,overwrite=overwrite)
        attr = {"unitsInSI":angstrom}
        galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        # Print progress if requested
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        # Optionally return dataset array
        if returnDataset:
            return equivalentWidth
        del equivalentWidth
        return



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
            luminosity = getLuminosity(galHDF5Obj,z,filter.replace("disk","total"))
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

