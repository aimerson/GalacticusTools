#! /usr/bin/env python

import os,fnmatch
import numpy as np
import pkg_resources



class Halpha(object):
    
    def __init__(self,dataset):        
        if fnmatch.fnmatch(dataset.lower(),"*colbert*"):
            self.dataset = "Colbert et al. (2013)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Colbert13_Halpha.dat")
            dtype = [("z",float),("log10L",float),("number",int),("phi",float),("phiCorr",float),("phiCorrErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.hubble = 0.7
        return

    def selectRedshift(self,z):
        data = None
        if self.dataset == "Colbert et al. (2013)":
            if z > 0.3 and z < 0.9:
                mask = self.data.z == 0.6
            if z > 0.9 and z < 1.5:
                mask = self.data.z == 1.2
            data = self.data[mask]            
        return data


