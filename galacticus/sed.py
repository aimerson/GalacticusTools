#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from galacticus.io import GalacticusHDF5
from galacticus.Luminosities import ergPerSecond
from galacticus.constants import erg,luminosityAB,jansky
from galacticus.constants import angstrom,megaParsec,Pi,speedOfLight


class GalacticusSED(object):
    
    def __init__(self,galObj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galacticusOBJ = galObj
        return

    def getSED(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get redshift
        redshift = float(fnmatch.filter(datasetName.split(":"),"z*")[0].replace("z",""))
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
        luminosities = []
        dummy = [luminosities.append(np.array(out["nodeData/"+topHatNames[i]])) for i in range(len(wavelengths))]
        sed = np.stack(np.copy(luminosities),axis=1)
        del dummy,luminosities
        sed = ergPerSecond(sed)
        # Convert units to microJanskys
        comDistance = self.galacticusOBJ.cosmology.comoving_distance(redshift)*megaParsec*100.0
        sed /= 4.0*Pi*comDistance**2
        frequency = speedOfLight/np.stack([wavelengths]*ngals)*angstrom
        sed = sed/np.copy(frequency)
        sed / jansky
        del frequency
        sed *= 1000.0
        return wavelengths,sed
        
        
        


        



