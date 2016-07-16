#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import pkg_resources


class Halpha(object):
    
    def __init__(self,dataset):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # List available datasets
        self.available = ["Colbert13","Sobral13"]
        # Load appropirate dataset
        if fnmatch.fnmatch(dataset.lower(),"*colbert*"):
            self.dataset = "Colbert et al. (2013)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Colbert13_Halpha.dat")
            dtype = [("z",float),("log10L",float),("number",int),("phi",float),("phiCorr",float),("phiCorrErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.hubble = 0.7
        elif fnmatch.fnmatch(dataset.lower(),"*sobral*"):
            self.dataset = "Sobral et al. (2013)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Sobral13_Halpha.dat")
            dtype = [("z",float),("log10L",float),("errlog10L",float),("number",int),("phiObs",float),("phiObsErr",float),\
                         ("phiCorr",float),("phiCorrErr",float),("volume",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
            self.hubble = 0.7
        else:
            raise ValueError(classname+"(): Dataset name not recognised! Currently available datasets: "+\
                                 ",".join(self.avaialble))
        return

    def selectRedshift(self,z):
        data = None
        if self.dataset == "Colbert et al. (2013)":
            mask = np.zeros(len(self.data.z),bool)
            if z > 0.3 and z < 0.9:
                mask = self.data.z == 0.6
            if z > 0.9 and z < 1.5:
                mask = self.data.z == 1.2
            data = self.data[mask]            
        if self.dataset == "Sobral et al. (2013)":
            mask = np.zeros(len(self.data.z),bool)
            if z > 0.35 and z < 0.45:
                mask = self.data.z == 0.4
            if z > 0.75 and z < 0.95:
                mask = self.data.z == 0.84
            if z > 1.35 and z < 1.55:
                mask = self.data.z == 1.47
            if z > 1.7 and z < 2.8:
                mask = self.data.z == 2.23
            data = self.data[mask]                    
        return data


