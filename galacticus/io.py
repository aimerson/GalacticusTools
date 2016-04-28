#! /usr/bin/env python

import os,sys
import numpy as np
import h5py

######################################################################################
# FILE I/O
######################################################################################


def check_readonly(func):
    def wrapper(self,*args,**kwargs):
        funcname = self.__class__.__name__+"."+func.__name__
        if self.read_only:
            raise IOError(funcname+"(): HDF5 file "+self.filename+" is READ ONLY!")
        return func(self,*args,**kwargs)
    return wrapper


class GalacticusHDF5(object):
    
    def __init__(self,*args,**kwargs):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "verbose" in kwargs.keys():
            self._verbose = kwargs["verbose"]
        else:
            self._verbose = False

        # Open file and store file object
        self.fileObj = h5py.File(*args)

        # Store file name and access mode
        self.filename = self.fileObj.filename
        if self._verbose:
            print(classname+"(): HDF5 file = "+self.filename)
        if self.fileObj.mode == "r":
            self.read_only = True
            if self._verbose:
                print(classname+"(): HDF5 opened in READ-ONLY mode")
        elif self.fileObj.mode == "r+":
            self.read_only = False

        # Store version information
        self.version = dict(self.fileObj["Version"].attrs)
        
        # Store build information
        self.build = dict(self.fileObj["Build"].attrs)

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

        return


    def close(self):
        self.fileObj.close()
        return


    def global_history(self,props=None,si=False):        
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


    def galaxies(self,props=None,z=None,SIunits=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select epoch closest to specified redshift
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        outstr = "Output"+str(self.outputs["iout"][iselect])
        out = self.fileObj["Outputs/"+outstr]
        if self._verbose:
            print(funcname+"(): Reading "+outstr+" (redshift = "+str(self.outputs.z[iselect])+")")
        # Set list of all available properties
        allprops = out["nodeData"].keys()
        # Get number of galaxies
        ngals = len(np.array(out["nodeData/"+allprops[0]]))
        if self._verbose:
            print(funcname+"(): Number of galaxies = "+str(ngals))
        # Construct datatype for galaxy properties to read
        if props is None:
            props = allprops        
        dtype = []
        for p in props:
            if p in allprops:
                dtype.append((p,out["nodeData/"+p].dtype))
            else:
                if p.lower() == "weight":
                    dtype.append((p.lower(),float))
        galaxies = np.zeros(ngals,dtype=dtype)
        # Extract galaxy properties
        for p in props:
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


