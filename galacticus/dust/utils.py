#! /usr/bin/env python

import sys,os,re,fnmatch
import numpy as np
from ..io import GalacticusHDF5
from ..constants import Pi,Parsec,massAtomic,massSolar,massFractionHydrogen
from ..EmissionLines import getLineNames,GalacticusEmissionLines
from ..Filters import GalacticusFilters


class DustProperties(object):
    
    def __init__(self):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name    
        # Compute and store optical depth normalisation
        self.localISMMetallicity = 0.02  # ... Metallicity in the local ISM.
        self.AV_to_EBV = 3.10            # ... (A_V/E(B-V); Savage & Mathis 1979)
        self.NH_to_EBV = 5.8e21          # ... (N_H/E(B-V); atoms/cm^2/mag; Savage & Mathis 1979)
        self.opticalDepthToMagnitudes = 2.5*np.log10(np.exp(1.0)) # Conversion factor from optical depth to magnitudes of extinction.
        hecto = 1.0e2
        self.opticalDepthNormalization = (1.0/self.opticalDepthToMagnitudes)*(self.AV_to_EBV/self.NH_to_EBV)
        self.opticalDepthNormalization *= (massFractionHydrogen/massAtomic)*(massSolar/(Parsec*hecto)**2)/self.localISMMetallicity
        # Create objects to store filter and emission line information
        self.emissionLinesClass = GalacticusEmissionLines()
        self.filtersDatabase = GalacticusFilters()
        return

    def effectiveWavelength(self,name,redshift=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if name in self.emissionLinesClass.getLineNames():
            # Emission line:
            effectiveWavelength = self.emissionLinesClass.getWavelength(name)
        else:
            # Photometric band
            effectiveWavelength = self.filtersDatabase.getEffectiveWavelength(name,verbose=verbose)
            if redshift is None:
                effectiveWavelength /= (1.0+float(redshift))
            if verbose:
                infoLine = "filter={0:s}  effectiveWavelength={1:s}".format(name,effectiveWavelength)
                print(funcname+"(): Photometric filter information:\n        "+infoLine)
        return effectiveWavelength
    
        
