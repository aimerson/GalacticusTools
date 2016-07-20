#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import pkg_resources


class Halpha(object):
    
    def __init__(self,dataset):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # List available datasets
        self.available = ["Gallego95","Shim09","Colbert13","Sobral13"]
        # Load appropirate dataset
        if fnmatch.fnmatch(dataset.lower(),"*gallego*"):
            self.dataset = "Gallego et al. (1995)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Gallego95_Halpha.dat")
            dtype = [("log10L",float),("phi",float),("phiErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.data.log10L = np.log10(self.data.log10L*1.0e40)
            self.hubble = 0.5
        elif fnmatch.fnmatch(dataset.lower(),"*shim*"):
            self.dataset = "Shim et al. (2009)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Shim09_Halpha.dat")
            dtype = [("z",float),("log10L",float),("phi",float),("phiErr",float),("number",int)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.data.phi *= 1.0e-3
            self.data.phiErr *= 1.0e-3
            self.hubble = 0.71
        elif fnmatch.fnmatch(dataset.lower(),"*colbert*"):
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
        if self.dataset == "Gallego et al. (1995)":
            if z > 0.1:
                mask = np.zeros(len(self.data.log10L),bool)
            else:
                mask = np.ones(len(self.data.log10L),bool)
            data = self.data[mask]            
        if self.dataset == "Shim et al. (2009)":
            mask = np.zeros(len(self.data.z),bool)
            if z > 0.7 and z < 1.4:
                mask = self.data.z == 1.05
            if z > 1.4 and z < 1.9:
                mask = self.data.z == 1.65
            data = self.data[mask]            
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
        if len(data.log10L) == 0:
            data = None
        return data






class PhotometricBand(object):

    def __init__(self,dataset,band):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.dataset = dataset
        self.band = band
        # Load appropriate LF dataset
        if fnmatch.fnmatch(dataset.lower(),"gama") or fnmatch.fnmatch(dataset.lower(),"driver*"):
            self.data = read_GAMA_LF(band)
        elif fnmatch.fnmatch(dataset.lower(),"2df*") or fnmatch.fnmatch(dataset.lower(),"cole*") or \
                fnmatch.fnmatch(data.lower(),"norberg*"):
            self.data = read_2dFGRS_LF(band)
        return 
        

def read_GAMA_LF(band):
    funcname = sys._getframe().f_code.co_name
    available = "J H K u g r i z Y NUV FUV".split()
    if band.lower() not in list(map(lambda x:x.lower(),available)):
        raise ValueError(funcname+"(): Band not recognised! Available bands are: "+",".join(available))
    ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Driver12_"+band+"_z0.dat")
    dtype = [("mag",float),("phi",float),("phiErr",float),("number",float)]
    data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
    data.phi /= 0.5
    data.phiErr /= 0.5
    return data
             
def read_2dFGRS_LF(band):
    funcname = sys._getframe().f_code.co_name
    available = "J K Ks bJ".split()
    if band.lower() not in list(map(lambda x:x.lower(),available)):
        raise ValueError(funcname+"(): Band not recognised! Available bands are: "+",".join(available))
    if band.lower() in list(map(lambda x:x.lower(),"k ks j".split())):
        ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Cole01_JK_z0.dat")
        dtype = [("mag",float),("phi",float),("phiErr",float)]
        if band.lower() == "j":
            usecols = [0,1,2]
        else:
            usecols = [0,3,4]
        data = np.loadtxt(ifile,dtype=dtype,usecols=usecols).view(np.recarray)
    else:
        ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Norberg02_bJ_z0.dat")
        dtype = [("mag",float),("phi",float),("phiErr",float),("sumWeight",float),("meanMag",float)]
        usecols = [1,0,2,3,4]
        data = np.loadtxt(ifile,dtype=dtype,usecols=usecols).view(np.recarray)
    return data
