#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import pkg_resources
from .analyticFits import SchechterLuminosities,SaundersLuminosities

class Halpha(object):
    
    def __init__(self,dataset,hubble=0.7):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.hubble = hubble

        # List available datasets
        self.available = ["Gallego95","Shim09","Colbert13","Sobral13","Gunawardhana13(GAMA)","Gunawardhana13(GAMA)"]
        # Load appropirate dataset
        if fnmatch.fnmatch(dataset.lower(),"*gallego*"):
            self.dataset = "Gallego et al. (1995)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Gallego95_Halpha.dat")
            dtype = [("log10L",float),("phi",float),("phiErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.data.log10L = np.log10(self.data.log10L*1.0e40)
            hubble = 0.5
            self.data.log10L = self.data.log10L + np.log10((hubble/self.hubble)**2)
            self.data.phi = self.data.phi*((self.hubble/hubble)**3)
            self.data.phiErr = self.data.phiErr*((self.hubble/hubble)**3)
        elif fnmatch.fnmatch(dataset.lower(),"*shim*"):
            self.dataset = "Shim et al. (2009)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Shim09_Halpha.dat")
            dtype = [("z",float),("log10L",float),("phi",float),("phiErr",float),("number",int)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            self.data.phi *= 1.0e-3
            self.data.phiErr *= 1.0e-3
            hubble = 0.71
            self.data.log10L = self.data.log10L + np.log10((hubble/self.hubble)**-2)
            self.data.phi = self.data.phi*((self.hubble/hubble)**3)
            self.data.phiErr = self.data.phiErr#*((self.hubble/hubble)**3)
        elif fnmatch.fnmatch(dataset.lower(),"*colbert*"):
            self.dataset = "Colbert et al. (2013)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Colbert13_Halpha.dat")
            dtype = [("z",float),("log10L",float),("number",int),("phi",float),("phiCorr",float),("phiCorrErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)            
            hubble = 0.7
            self.data.log10L = self.data.log10L + np.log10((hubble/self.hubble)**2)
            self.data.phi = self.data.phi*((self.hubble/hubble)**3)
            self.data.phiCorr = self.data.phiCorr*((self.hubble/hubble)**3)
            self.data.phiCorrErr = self.data.phiCorrErr*((self.hubble/hubble)**3)
        elif fnmatch.fnmatch(dataset.lower(),"*sobral*"):
            self.dataset = "Sobral et al. (2013)"
            ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Sobral13_Halpha.dat")
            dtype = [("z",float),("log10L",float),("errlog10L",float),("number",int),("phiObs",float),("phiObsErr",float),\
                         ("phiCorr",float),("phiCorrErr",float),("volume",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
            hubble = 0.7
            self.data.log10L -= 2.0/5.0
            self.data.log10L = self.data.log10L + np.log10((hubble/self.hubble)**-2)
            self.data.errlog10L = self.data.errlog10L 
            self.data.phiObs = self.data.phiObs + np.log10((self.hubble/hubble)**3)
            self.data.phiObsErr = self.data.phiObsErr
            self.data.phiCorr = self.data.phiCorr + np.log10((self.hubble/hubble)**3)
            self.data.phiCorrErr = self.data.phiCorrErr
            self.data.volume *= 1.0e4            
            self.data.volume *= ((hubble/self.hubble)**3)
        elif fnmatch.fnmatch(dataset.lower(),"*gunawardhana*"):
            self.dataset = "Gunawardhana et al. (2013)"
            if "sdss" in dataset.lower():
                ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Gunawardhana13_SDSS_Halpha.dat")
                self.dataset = self.dataset + " [SDSS]"
            elif "gama" in dataset.lower():
                ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Gunawardhana13_GAMA_Halpha.dat")
                self.dataset = self.dataset + " [GAMA]"
            else:
                raise ValueError(classname+"(): Gunawardhana et al. data available for two datasets! Specify either GAMA or SDSS in dataset name!")
            dtype = [("z",float),("log10L",float),("logphi",float),("logphiNegErr",float),("logphiPosErr",float)]
            self.data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
            self.data.log10L += 7.0  # Convert from Watts to ergs/s
            hubble = 0.7
            self.data.log10L += np.log10((hubble/self.hubble)**2)
            self.data.logphi += np.log10((self.hubble/hubble)**3)
            self.data.logphiNegErr += np.log10((self.hubble/hubble)**3)
            self.data.logphiPosErr += np.log10((self.hubble/hubble)**3)            
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
        if fnmatch.fnmatch(self.dataset,"Gunawardhana et al. (2013)*"):
            mask = np.zeros(len(self.data.z),bool)
            if z < 0.1:
                mask = self.data.z == 0.05
            if z > 0.1 and z < 0.15:
                mask = self.data.z == 0.125
            if z > 0.17 and z < 0.24:
                mask = self.data.z == 0.205
            if z > 0.24 and z < 0.34:
                mask = self.data.z == 0.29
            data = self.data[mask]                            
        if len(data.log10L) == 0:
            data = None
        return data


    def Schechter(self,z):
        object = None
        if self.dataset == "Gallego et al. (1995)":
            if z < 0.1:
                hubble = 0.7 # Rescaled values from Geach et al (2010)
                Lstar = 10**41.87                
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.78
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.3,Lstar,phistar)
        if self.dataset == "Colbert et al. (2013)":
            hubble = 0.7
            if z > 0.3 and z < 0.9:
                Lstar = 10**41.72                
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.51
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.27,Lstar,phistar)
            if z > 0.9 and z < 1.5:
                Lstar = 10**42.18
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.70
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.43,Lstar,phistar)
        if self.dataset == "Sobral et al. (2013)":
            hubble = 0.7
            if z >= 0.39 and z <= 0.41:
                Lstar = 10**41.95                
                Lstar *= 10**-0.4
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-3.12
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.75,Lstar,phistar)
            if z >= 0.82 and z <= 0.86:
                Lstar = 10**42.25
                Lstar *= 10**-0.4                                
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.47
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.56,Lstar,phistar)                
            if z >= 1.45 and z <= 1.49:
                Lstar = 10**42.56                
                Lstar *= 10**-0.4                
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.61
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.62,Lstar,phistar)                                
            if z >= 2.21 and z <= 2.25:
                Lstar = 10**42.87                
                Lstar *= 10**-0.4                
                Lstar *= ((hubble/self.hubble)**2)
                phistar = 10**-2.78
                phistar *= ((self.hubble/hubble)**3)
                object = SchechterLuminosities(-1.59,Lstar,phistar)                                
        return object








class PhotometricBand(object):

    def __init__(self,dataset,band):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.dataset = dataset
        self.band = band
        # Load appropriate LF dataset
        if fnmatch.fnmatch(dataset.lower(),"gama") or fnmatch.fnmatch(dataset.lower(),"driver*"):
            self.data,self.reference,self.survey = read_GAMA_LF(band)
        elif fnmatch.fnmatch(dataset.lower(),"2df*") or fnmatch.fnmatch(dataset.lower(),"cole*") or \
                fnmatch.fnmatch(dataset.lower(),"norberg*"):
            self.data,self.reference,self.survey = read_2dFGRS_LF(band)
        elif fnmatch.fnmatch(dataset.lower(),"2mass") or fnmatch.fnmatch(dataset.lower(),"kochanek*"):
            self.data,self.reference,self.survey = read_2MASS_LF(band)
        elif fnmatch.fnmatch(dataset.lower(),"6df*") or fnmatch.fnmatch(dataset.lower(),"jones*"):
            self.data,self.reference,self.survey = read_6dFGS_LF(band)
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
    return data,"Driver et al. (2012)","GAMA"

def read_2MASS_LF(band):
    funcname = sys._getframe().f_code.co_name
    available = "K".split()
    if band.lower() not in list(map(lambda x:x.lower(),available)):
        raise ValueError(funcname+"(): Band not recognised! Available bands are: "+",".join(available))
    ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Kochanek01_"+band+"_z0.dat")
    dtype = [("mag",float),("logphi",float),("logphiErr",float)]
    data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
    data.mag += 1.85
    return data,"Kochanek et al. (2001)","2MASS"


def read_6dFGS_LF(band):
    funcname = sys._getframe().f_code.co_name
    available = "K".split()
    if band.lower() not in list(map(lambda x:x.lower(),available)):
        raise ValueError(funcname+"(): Band not recognised! Available bands are: "+",".join(available))
    ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Jones06_"+band+"_z0.dat")
    dtype = [("mag",float),("logphi",float),("logphiPosErr",float),("logphiNegErr",float),("number",int)]
    data = np.loadtxt(ifile,dtype=dtype,usecols=range(len(dtype))).view(np.recarray)
    data.mag += 1.85
    data.logphi -= np.log10(0.25)
    data.logphiNegErr = np.fabs(data.logphiNegErr) 
    return data,"Jones et al. (2006)", "6dFGS"
             
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
        if band.lower() == "j":
            data.mag += 0.91
        else:
            data.mag += 1.85        
        reference = "Cole et al. (2001)"
    else:
        ifile = pkg_resources.resource_filename(__name__,"../../data/LuminosityFunctions/Norberg02_bJ_z0.dat")
        dtype = [("mag",float),("phi",float),("phiErr",float),("sumWeight",float),("meanMag",float)]
        usecols = [1,0,2,3,4]
        data = np.loadtxt(ifile,dtype=dtype,usecols=usecols).view(np.recarray)
        data.mag -= 0.09
        reference = "Norberg et al. (2002)"
    return data,reference,"2dFGRS"
