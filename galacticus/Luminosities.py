#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .galaxyProperties import DatasetClass
from .constants import luminositySolar,erg

def ergPerSecond(luminosity):    
    luminosity = np.log10(luminosity)
    luminosity += np.log10(luminositySolar)
    luminosity -= np.log10(erg)
    luminosity = 10.0**luminosity
    return luminosity

class LuminosityClass(DatasetClass):
    
    def __init__(self,datasetName=None,luminosity=None,\
                     redshift=None,outputName=None):
        super(LuminosityClass,self).__init__(datasetName=datasetName,redshift=redshift,\
                                                 outputName=outputName)
        self.luminosity = luminosity
        return
