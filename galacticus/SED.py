#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
from scipy.interpolate import interp1d
from galacticus.io import GalacticusHDF5
from galacticus.EmissionLines import GalacticusEmissionLines
from galacticus.Luminosities import ergPerSecond
from galacticus.constants import erg,luminosityAB,luminositySolar,jansky
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
        
    
    def addEmissionLines(self,wavelengths,luminosities,MATCH,selectionMask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Extract necessary information from Regex dataset match
        component = MATCH.group(1)
        frame = MATCH.group(3)
        redshift = MATCH.group(4)
        dust = MATCH.group(5)
        if dust is None:
            dust = ""
        option = MATCH.group(6)
        if option is None:
            option = ""
        # Check for any emission lines inside wavelength range
        lines = np.array(self.EmissionLines.getLineNames())
        if frame == "rest":
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name) for name in lines])
        else:
            effectiveWavelengths = np.array([self.EmissionLines.getWavelength(name,redshift=float(redshift)) for name in lines])
        
        emissionLinesInRange = np.logical_and(effectiveWavelengths>=wavelengths.min(),effectiveWavelengths<=wavelengths.max())
        # If no emission lines in range, return original wavelengths and luminosities
        if not any(emissionLinesInRange):
            return wavelengths,luminosities        
        # Select names and wavelengths of emission lines in range
        wavelengthsInRange = effectiveWavelengths[emissionLinesInRange]
        linesInRange = lines[emissionLinesInRange]
        if self._verbose:
            report = funcname+"(): Following emission lines are inside SED wavelength range:"
            report = report + "\n     " + "\n     ".join(list(linesInRange))
            print(report)
        # Construct interpolation object
        interpLuminosity = interp1d(np.copy(wavelengths),np.copy(luminosities),axis=1)        
        # Add emission lines wavelengths into wavelengths of SED
        buffer = 1.0              
        wavelengths = list(np.copy(wavelengths)) + list(np.copy(wavelengthsInRange)) + \
            list(np.copy(wavelengthsInRange)+buffer) + list(np.copy(wavelengthsInRange)-buffer)
        wavelengths = np.sort(np.unique(np.array(wavelengths)))
        # Interpolate to get continuum luminosity at each wavelength
        luminosities = interpLuminosity(wavelengths)
        # Get output for redshift
        out = self.galacticusOBJ.selectOutput(float(redshift))
        # Add emission lines in as delta functions
        def _addLuminosity(lineName,wavelength):            
            iarg = np.argwhere(np.fabs(wavelengths-wavelength)<buffer)[0][0]
            lineDatasetName = component+"LineLuminosity:"+lineName+":"+frame+":z"+redshift+dust+option
            if lineDatasetName not in out["nodeData"].keys():
                print("WARNING! "+funcname+"(): Cannot locate '"+lineDatasetName+"' for inclusion in SED. Will be skipped.")
            else:                
                lineLuminosity = np.array(out["nodeData/"+lineDatasetName])
                if selectionMask is not None:
                    lineLuminosity = lineLuminosity[selectionMask]
                np.place(lineLuminosity,lineLuminosity<0.0,0.0)
                luminosities[:,iarg] += lineLuminosity*(luminositySolar/luminosityAB)
        dummy = [_addLuminosity(linesInRange[i],wavelengthsInRange[i]) for i in range(len(linesInRange))]
        del dummy
        return wavelengths,luminosities        
        


    def ergPerSecond(self,luminosity):
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
            wavelengths,sed = self.addEmissionLines(wavelengths,sed,MATCH,selectionMask=selectionMask)            
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
        
        
        


        



