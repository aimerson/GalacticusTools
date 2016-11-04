#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
from .hdf5 import HDF5
from .utils.datatypes import getDataType
from .cosmology import Cosmology

######################################################################################
# FILE I/O
######################################################################################

class GalacticusHDF5(HDF5):
    
    def __init__(self,*args,**kwargs):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        
        # Initalise HDF5 class
        super(GalacticusHDF5, self).__init__(*args,**kwargs)

        # Store version information
        self.version = dict(self.fileObj["Version"].attrs)
        
        # Store build information if available
        try:            
            self.build = dict(self.fileObj["Build"].attrs)
        except KeyError:
            self.build = None

        # Store parameters
        self.parameters = dict(self.fileObj["Parameters"].attrs)
        self.parameters_parents = { k:"parameters" for k in self.fileObj["Parameters"].attrs.keys()}
        for k in self.fileObj["Parameters"]:
            if len(self.fileObj["Parameters/"+k].attrs.keys())>0:
                d = dict(self.fileObj["Parameters/"+k].attrs)                
                self.parameters.update(d)
                d = { a:k for a in self.fileObj["Parameters/"+k].attrs.keys()}
                self.parameters_parents.update(d)

        # Store output epochs
        Outputs = self.fileObj["Outputs"]
        nout = len(Outputs.keys())
        self.outputs = np.zeros(nout,dtype=[("iout",int),("a",float),("z",float)])
        for i,out in enumerate(Outputs.keys()):
            self.outputs["iout"][i] = int(out.replace("\n","").replace("Output",""))
            a = float(Outputs[out].attrs["outputExpansionFactor"])
            self.outputs["a"][i] = a
            self.outputs["z"][i] = (1.0/a) - 1.0
        self.outputs = self.outputs.view(np.recarray)

        # Store cosmology object
        
        self.cosmology = Cosmology(omega0=float(self.parameters["OmegaMatter"]),\
                                       lambda0=float(self.parameters["OmegaDarkEnergy"]),\
                                       omegab=float(self.parameters["OmegaBaryon"]),\
                                       h0=float(self.parameters["HubbleConstant"])/100.0,\
                                       sigma8=float(self.parameters["sigma_8"]),\
                                       ns=float(self.parameters["index"]))
        
        return
    
    def globalHistory(self,props=None,si=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        globalHistory = self.fileObj["globalHistory"]
        allprops = globalHistory.keys() + ["historyRedshift"]
        if props is None:
            props = allprops 
        else:
            props = set(props).intersection(allprops)            
        epochs = len(np.array(globalHistory["historyExpansion"]))
        dtype = np.dtype([ (str(p),np.float) for p in props ])
        history = np.zeros(epochs,dtype=dtype)
        if self._verbose:
            if si:
                print("WARNING! "+funcname+"(): Adopting SI units!")
            else:
                print("WARNING! "+funcname+"(): NOT adopting SI units!")        
        for p in history.dtype.names:
            if p is "historyRedshift":
                history[p] = np.copy((1.0/np.array(globalHistory["historyExpansion"]))-1.0)
            else:
                history[p] = np.copy(np.array(globalHistory[p]))
                if si:
                    if "unitsInSI" in globalHistory[p].attrs.keys():
                        unit = globalHistory[p].attrs["unitsInSI"]
                        history[p] = history[p]*unit
        return history.view(np.recarray)

    def selectOutput(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select epoch closest to specified redshift
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        outstr = "Output"+str(self.outputs["iout"][iselect])
        if self._verbose:
            print(funcname+"(): Nearest output is "+outstr+" (redshift = "+str(self.outputs.z[iselect])+")")
        return self.fileObj["Outputs/"+outstr]
        
    def availableDatasets(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        out = self.selectOutput(z)
        return map(str,out["nodeData"].keys())

    def readGalaxies(self,z,props=None,SIunits=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select epoch closest to specified redshift
        out = self.selectOutput(z)
        # Set list of all available properties
        allprops = self.availableDatasets(z)
        # Get number of galaxies
        ngals = len(np.array(out["nodeData/"+allprops[0]]))
        if self._verbose:
            print(funcname+"(): Number of galaxies = "+str(ngals))
        # Read all properties if not specified
        if props is None:
            props = allprops        
        # Check for properties not already calculated
        if len(list((set(props).difference(allprops))))>0:
            self.calculateProperties(list(set(props).difference(allprops)),z,overwrite=False)
        # Construct datatype for galaxy properties to read        
        dtype = []
        for p in props:
            if len(fnmatch.filter(allprops,p))>0:
                matches = fnmatch.filter(allprops,p)
                dtype = dtype + [ (str(m),getDataType(out["nodeData/"+m])) for m in matches ]                
            else:
                if p.lower() == "weight":
                    dtype.append((p.lower(),float))
        galaxies = np.zeros(ngals,dtype=dtype)
        # Extract galaxy properties
        for p in galaxies.dtype.names:
            if p in allprops:
                galaxies[p] = np.copy(np.array(out["nodeData/"+p]))
                if SIunits:
                    if "unitsInSI" in out["nodeData/"+p].attrs.keys():
                        unit = out["nodeData/"+p].attrs["unitsInSI"]
                        galaxies[p] *= unit
            else:
                if p.lower() == "weight":
                    cts = np.array(out["mergerTreeCount"])
                    wgt = np.array(out["mergerTreeWeight"])
                    galaxies[p.lower()] = np.copy(np.repeat(wgt,cts))
                    del cts,wgt            
        return galaxies

        
    def calculateProperties(self,galaxyProperties,z,overwrite=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        

        # Galaxy inclination
        if "inclination" in galaxyProperties:            
            from .Inclination import Get_Inclination
            if self._verbose:
                print(funcname+"(): calculating inclination (z="+str(z)+")")
            data = Get_Inclination(self,z,overwrite=overwrite)
            del data
        # Total stellar mass
        if "massStellar" in galaxyProperties:
            from .StellarMass import Get_StellarMass
            if self._verbose:
                print(funcname+"(): calculating massStellar (z="+str(z)+")")
            data = Get_StellarMass(self,z,overwrite=overwrite)
            del data
        # Total star formation rate
        if "starFormationRate" in galaxyProperties:
            from .StellarMass import Get_StarFormationRate
            data = Get_StarFormationRate(self,z,overwrite=overwrite)
            if self._verbose:
                print(funcname+"(): calculating starFormationRate (z="+str(z)+")")
            del data

        return



