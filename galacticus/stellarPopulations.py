#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from .hdf5 import HDF5


class stellarPopulationSynthesisModel(HDF5):

    def __init__(self,*args,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(stellarPopulationSynthesisModel, self).__init__(*args,**kwargs)
        # Load spectra
        self.wavelengths = np.array(self.fileObj["wavelengths"])
        self.metallicities = np.array(self.fileObj["metallicities"])
        self.ages = np.array(self.fileObj["ages"])
        self.spectra = np.array(self.fileObj["spectra"])
        # Load IMF
        self.mass = np.array(self.fileObj["initialMassFunction/mass"])
        self.imf = np.array(self.fileObj["initialMassFunction/initialMassFunction"])
        return

    def wavelengthInterval(self,wavelength):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not np.logical_and(wavelength>=self.wavelengths[0],wavelength<=self.wavelengths[-1]):
            raise IndexError(funcname+"(): specified wavelength is outside wavelength range of SPS model.")        
        diff = self.wavelengths - wavelength
        mask = diff > 0.0
        upp = self.wavelengths[mask].min()
        mask = diff <= 0.0
        low = self.wavelengths[mask].max()
        return upp-low
    
        
