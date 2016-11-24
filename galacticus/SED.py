#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from scipy.interpolate import interp1d
from galacticus.io import GalacticusHDF5
from galacticus.EmissionLines import GalacticusEmissionLines
from galacticus.Luminosities import ergPerSecond
from galacticus.constants import erg,luminosityAB,luminositySolar,jansky,kilo
from galacticus.constants import angstrom,megaParsec,Pi,speedOfLight,centi


class GalacticusSED(object):
    
    def __init__(self,galObj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galacticusOBJ = galObj
        self.EmissionLines = GalacticusEmissionLines()
        self._verbose = verbose
        return


    def availableSEDs(self,redshift):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get list of all top hat filers at this redshift
        allTopHats = fnmatch.filter(self.galacticusOBJ.availableDatasets(redshift),"*LuminositiesStellar:topHat*")
        # For each top hat filter remove wavelength and rename 'LuminositiesStellar' to 'SED'
        allSEDs = []
        for topHat in allTopHats:
            sed = topHat.split(":")
            resolution = sed[1].split("_")[-1]
            sed[1] = resolution
            sed[0] = sed[0].replace("LuminositiesStellar","SED")
            sed = ":".join(sed)
            allSEDs.append(sed)
        # Return only unique list to avoid duplicates
        return list(np.unique(np.array(allSEDs)))


    def getSEDLuminosity(self,topHatName,redshift,selectionMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get output for redshift
        out = self.galacticusOBJ.selectOutput(redshift)
        # Get stellar luminosity
        luminosity = np.array(out["nodeData/"+topHatName])
        if selectionMask is not None:
            luminosity = luminosity[selectionMask]
        return luminosity
        

    def ergPerSecond(self,luminosity):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        luminosity = np.log10(luminosity)
        luminosity += np.log10(luminosityAB)
        luminosity -= np.log10(erg)
        luminosity = 10.0**luminosity
        return luminosity
        
    def getSED(self,datasetName,selectionMask=None,includeEmissionLines=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset name correspnds to an SED
        MATCH = re.search(r"^(disk|spheroid|total)SED:([^:]+):([^:]+):z([\d\.]+)(:dust[^:]+)?(:recent)?",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Extract necessary information
        redshift = float(MATCH.group(4))
        frame = MATCH.group(3)
        # Identify top hat filters
        search = datasetName.replace("SED:","LuminositiesStellar:topHat_*_")    
        topHatNames = fnmatch.filter(list(map(str,self.galacticusOBJ.availableDatasets(redshift))),search)
        # Extract wavelengths and then sort wavelengths and top hat names according to wavelength
        wavelengths = np.array([float(topHat.split("_")[1]) for topHat in topHatNames])
        isort = np.argsort(wavelengths)
        wavelengths = wavelengths[isort]
        topHatNames = list(np.array(topHatNames)[isort])
        # Construct 2D array of galaxy SEDs         
        out = self.galacticusOBJ.selectOutput(redshift)
        ngals = len(np.array(out["nodeData/"+topHatNames[0]]))
        if selectionMask is None:
            selectionMask = np.ones(ngals,dtype=bool)
        else:
            if len(selectionMask) != ngals:
                raise ValueError(funcname+"(): specified selection mask does not have same shape as datasets!")
        luminosities = []
        dummy = [luminosities.append(self.getSEDLuminosity(topHatNames[i],redshift,selectionMask=selectionMask))\
                     for i in range(len(wavelengths))]
        sed = np.stack(np.copy(luminosities),axis=1)
        del dummy,luminosities
        # Introduce emission lines
        if includeEmissionLines:
            LINES = EmissionLineProfiles(self.galacticusOBJ,frame,redshift,wavelengths,selectionMask=selectionMask,verbose=self._verbose)
            if not len(LINES.emissionLinesInRange) == 0:
                sed += LINES.sumLineProfiles(MATCH,type='gaussian')
        # Convert units to microJanskys
        sed = self.ergPerSecond(sed)
        comDistance = self.galacticusOBJ.cosmology.comoving_distance(redshift)*megaParsec/centi
        sed /= 4.0*Pi*comDistance**2
        frequency = speedOfLight/np.stack([wavelengths]*ngals)*angstrom
        sed = sed/np.copy(frequency)
        sed /= jansky
        del frequency
        sed *= 1.0e6
        return wavelengths,sed
        
        
        

class EmissionLineProfiles(object):
    
    def __init_(self,galObj,frame,redshift,wavelengths,selectionMask=None,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store GalacticusHDF5 object
        self.galacticusOBJ = galObj
        # Store redshift
        self.redshift = redshift
        # Store frame
        self.frame = frame
        # Store wavelengths
        self.sedWavelengths = wavelengths
        # Crete emission lines object
        self.EmissionLines = GalacticusEmissionLines()        
        # Check for any emission lines inside wavelength range
        lines = np.array(self.EmissionLines.getLineNames())
        if self.frame == "rest":
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name) for name in lines])
        else:
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name,redshift=float(redshift)) for name in lines])
        self.emissionLinesInRange = np.logical_and(effectiveWavelengths>=self.sedWavelengths.min(),effectiveWavelengths<=sedWavelengths.max())
        # Exit here if no lines in range
        if len(self.emissionLinesInRange) == 0:
            return
        # Store wavelengths the lines appear at
        self.wavelengthsInRange = effectiveWavelengths[emissionLinesInRange]
        # Store selection mask to apply to galaxies
        self.selectionMask = selectionMask
        # Store redshift output
        self.OUT = self.galacticusOBJ.selectOutput(redshift)
        # Create 2D array to store sum of line profiles 
        ngals = len(np.array(self.OUT["nodeData/nodeIndex"]))
        self.profileSum = np.zeros(ngals,len(self.wavelengths))
        return

    
    def sumLineProfiles(self,MATCH,profile='gaussian',lineWidth='fixed',fixedWidth=200.0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Sum profiles
        dummy = [self.addLine(lineName,MATCH,profile=profile,lineWidth=lineWidth,fixedWidth=fixedWidth)\
                     for lineName in self.emissionLinesInRange]
        # Return sum
        return self.profileSum


    def addLine(self,lineName,MATCH,profile='gaussian',lineWidth='fixed',fixedWidth=200.0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract dataset information
        component = MATCH.group(1)
        frame = MATCH.group(3)
        redshift = MATCH.group(4)
        dust = MATCH.group(5)
        if dust is None:
            dust = ""
        option = MATCH.group(6)
        if option is None:
            option = ""
        # Get dataset name
        lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+dust+option
        if lineDatasetName not in out["nodeData"].keys():
            print("WARNING! "+funcname+"(): Cannot locate '"+lineDatasetName+"' for inclusion in SED. Will be skipped.")
            return
        # Get wavelength line is found at
        iline = self.linesInRange.index(lineName)
        lineWavelength = self.wavelengthsInRange[iline]
        # Extract line luminosity
        lineLuminosity = np.array(out["nodeData/"+lineDatasetName])
        if selectionMask is not None:
            lineLuminosity = lineLuminosity[selectionMask]
        np.place(lineLuminosity,lineLuminosity<0.0,0.0)
        lineLuminosity *= (luminositySolar/luminosityAB)        
        # Compute FWHM (use fixed line width in km/s)
        FWHM = self.getFWHM(lineName)
        # Compute line profile
        if fnmatch.fnmatch(profile.lower(),"gaus*"):
            self.profileSum += self.gaussian(lineWavelength,lineLuminosity,FWHM)
        else:
            raise ValueError(funcname+"(): line profile must be Gaussian! Other profiles not yet implemented!")
        return 
        

    def getFWHM(self,lineName,lineWidth='fixed',fixedWidth=200.0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract rest wavelength
        restWavelength = self.EmissionLines.getWavelength(lineName)
        # Compute FWHM
        if lineWidth.lower() == "fixed":
            FWHM = restWavelength*(fixedWidth/(speedOfLight/kilo))
        else:
            raise ValueError(funcname+"(): line width method must be 'fixed'! Other methods not yet implemented!")
        return FWHM
    

    def gaussian(self,lineWavelength,lineLuminosity,FWHM):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Compute sigma for Gaussian
        sigma = FWHM/2.355
        # Compute amplitude for Gaussian
        amplitude = np.concatenate([lineLuminosity]*len(self.sedWavelengths)).reshape(len(lineLuminosity),-1)
        amplitude /= (sigma*np.sqrt(2.0*Pi))
        # Compute luminosity
        wavelengths = np.concatenate([self.sedWavelengths]*len(lineLuminosity)).reshape(-1,len(self.sedWavelengths))
        luminosity = amplitude*np.exp(-((wavelengths-lineWavelength)**2)/(2.0*(sigma**2)))
        return luminosity

