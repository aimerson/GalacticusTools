#! /usr/bin/env python

import os,sys,fnmatch
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

    ###############################################################################
    # READING DATASETS
    ###############################################################################
    
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

   def ls_datasets(self,hdfdir):
        objs = self.lsdir(hdfdir,recursive=False)
        dsets = []
        def _is_dataset(obj):
            return isinstance(self.fileObj[hdfdir+"/"+obj],h5py.Dataset)
        return filter(_is_dataset,objs)

    def selectOutput(self,z):
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
        

    def readDataset(self,hdfdir,recursive=False,required=None,exit_if_missing=True):
        """
        readDataset(): Read one or more HDF5 datasets.
        
        USAGE: data = GalacticusHDF5.readDataset(hdfdir,[recursive],[required],[exist_if_missing])

        Inputs:
        
               hdfdir    : Path to dataset or group of datasets to read.
               recursive : If reading HDF5 group, read recursively down subgroups.
                           (Default = False)
               required  : List of names of datasets to read. If required=None, will read
                           all datasets. (Default = None).
               exit_if_missing : Will raise error and exit if any of names in 'required'
                                 are missing. (Default = True).

        Outputs:
                data : Dictionary of datasets (stored as Numpy arrays).

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        data = {}
        if isinstance(self.fileObj[hdfdir],h5py.Dataset):
            # Read single dataset
            if hdfdir not in self.fileObj:
                raise KeyError(funcname+"(): "+hdfdir+" not found in HDF5 file!")
            name = hdfdir.split("/")[-1]
            data[str(name)] = np.array(self.fileObj[hdfdir])
        elif isinstance(self.fileObj[hdfdir],h5py.Group):
            # Read datasets in group 
            # i) List datasets (recursively if specified)
            if recursive:
                objs = self.lsdir(hdfdir,recursive=recursive)
            else:
                objs = self.ls_datasets(hdfdir)
            if required is not None:
                missing = list(set(required).difference(objs))
                if len(missing) > 0:
                    dashed = "-"*10
                    err = dashed+"\n"+funcname+"(): Following keys are missing from "+\
                        hdfdir+":\n     "+"\n     ".join(missing)+"\n"+dashed
                    print(err)
                    raise KeyError(funcname+"(): Some required keys cannot be found in "+hdfdir+"!")
                objs = list(set(objs).intersection(required))
            # ii) Store in dictionary
            def _store_dataset(obj):
                data[str(obj)] = np.array(self.fileObj[hdfdir+"/"+obj])
            map(_store_dataset,objs)
        return data


    def readGalaxies(self,props=None,z=None,SIunits=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select epoch closest to specified redshift
        out = self.selectOutput(z)
        # Set list of all available properties
        allprops = self.availableDatasets(z)
        # Get number of galaxies
        ngals = len(np.array(out["nodeData/"+allprops[0]]))
        if self._verbose:
            print(funcname+"(): Number of galaxies = "+str(ngals))
        # Construct datatype for galaxy properties to read
        if props is None:
            props = allprops        
        dtype = []
        for p in props:
            if len(fnmatch.filter(allprops,p))>0:
                matches = fnmatch.filter(allprops,p)
                dtype = dtype + [ (str(m),out["nodeData/"+m].dtype) for m in matches ]                
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


    ###############################################################################
    # GROUPS FUNCTIONS
    ###############################################################################

    @check_readonly
    def mkdir(self,hdfdir):
        """
        GalcaticusHDF5.mkdir(): create HDF5 group with specified path.

        USAGE: HDF5.mkdir(dir)
              
             Input: dir -- path to HDF5 group.
        """
        if hdfdir not in self.fileObj:
            g = self.fileObj.create_group(hdfdir)
        return

    @check_readonly
    def rmdir(self,hdfdir):
        """
        GalcticusHDF5.mkdir(): remove HDF5 group at specified path.
        
        USAGE: HDF5.rmdir(dir)
            
            Input: dir -- path to HDF5 group.
        """
        if hdfdir in self.fileObj:
            del self.fileObj[hdfdir]
        return
   
    @check_readonly
    def cpdir(self,srcfile,srcdir,dstdir=None):
        """
        GalcticusHDF5.cpdir(): copy HDF5 group with specified path from
        specified file.

        USAGE: HDF5.cpdir(srcfile,srcdir,[dstdir])

              Input: srcfile -- Path to source HDF5 file.  
                     srcdir  -- Path to source HDF5 group inside source file.
                     dstdir  -- Path to group to store copy of source group.  
                               Default = srcdir.


              Note that this function will create in the current file
              a parent group with the same path as the parent group of
              the source group in the source file.

        """
        # Open second file and get path to group that we want to copy
        fileObj = h5py.File(srcfile,"r")
        group_path = fileObj[srcdir].parent.name
        # Create same parent group in current file
        group_id = self.fileObj.require_group(group_path)
        # Set name of new group
        if dstdir is None:
            dstdir = srcdir
        fileObj.copy(srcdir,group_id,name=dstdir)
        fileObj.close()
        return

    def lsdir(self,hdfdir,recursive=False):
        ls = []
        thisdir = self.fileObj[hdfdir]
        if recursive:
            def _append_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    ls.append(name)
            thisdir.visititems(_append_item)
        else:
            ls = thisdir.keys()
        return ls

    ###############################################################################
    # WRITING DATASETS
    ###############################################################################
    
    @check_readonly
    def addDataset(self,hdfdir,name,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkdir(hdfdir)
        g = self.fileObj[hdfdir]
        # Write data to group
        value = np.copy(data)
        if name in g.keys():
            write_key = False
            if append:
                value = np.append(np.copy(g[name]),value)
                del g[name]
                write_key = True
            if overwrite:
                del g[name]
                write_key = True
        else:
            write_key = True
        if write_key:
            dset = g.create_dataset(name,data=value,maxshape=maxshape,\
                                        chunks=chunks,compression=compression,\
                                        compression_opts=compression_opts,**kwargs)
        del value
        return


    @check_readonly
    def addDatasets(self,hdfdir,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkdir(hdfdir)
        g = self.fileObj[hdfdir]
        # Write data to group
        for n in data.dtype.names:
            self.addDataset(hdfdir,n,data[n],append=append,overwrite=overwrite,\
                                maxshape=maxshape,chunks=chunks,compression=compression,\
                                compression_opts=compression_opts,**kwargs)
        return
    
