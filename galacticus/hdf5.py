#! /usr/bin/env python

import sys
import h5py
import numpy as np

def readonlyWrapper(func):
    def wrapper(self,*args,**kwargs):               
        funcname = self.__class__.__name__+"."+func.__name__
        if self.read_only:
            raise IOError(funcname+"(): HDF5 file "+self.filename+" is READ ONLY!")        
        return func(self,*args,**kwargs)
    return wrapper

class HDF5(object):
    
    def __init__(self,*args,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name

        self.fileObj = h5py.File(*args)
        if "verbose" in kwargs.keys():
            self._verbose = kwargs["verbose"]
        else:
            self._verbose = False
        self.filename = self.fileObj.filename        
        if self._verbose:
            print(classname+"(): HDF5 file = "+self.filename)
        if self.fileObj.mode == "r":
            self.read_only = True
            if self._verbose:
                print(classname+"(): HDF5 opened in READ-ONLY mode")
        elif self.fileObj.mode == "r+":
            self.read_only = False
        return
    
    def close(self):
        self.fileObj.close()
        return

    ##############################################################################
    # GROUPS
    ##############################################################################
    
    @readonlyWrapper
    def mkGroup(self,hdfdir):        
        """
        HDF5.mkGroup(): create HDF5 group with specified path.
        
        USAGE:  HDF5.mkdir(dir)

              Input: dir -- path to HDF5 group.       
        """
        if hdfdir not in self.fileObj:
            g = self.fileObj.create_group(hdfdir)
        return

    
    @readonlyWrapper
    def rmGroup(self,hdfdir):
        """
        HDF5.rmGroup(): remove HDF5 group at specified path.
        
        USAGE:  HDF5.rmdir(dir)

              Input: dir -- path to HDF5 group.       
        """
        if hdfdir in self.fileObj:
            del self.fileObj[hdfdir]
        return


    @readonlyWrapper
    def cpGroup(self,srcfile,srcdir,dstdir=None):        
        """
        HDF5.cpGroup(): copy HDF5 group with specified path from specified file.
        
        USAGE:  HDF5.cpGroup(srcfile,srcdir,[dstdir])

              Input: srcfile -- Path to source HDF5 file.       
                     srcdir -- Path to source HDF5 group inside source file.
                     dstdir -- Path to group to store copy of source group. 
                               Default = srcdir.

              Note that this function will create in the current file a parent 
              group with the same path as the parent group of the source group
              in the source file.
                                              
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

    
    def lsGroups(self,hdfdir,recursive=False):
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

    ##############################################################################
    # DATASETS
    ##############################################################################
    
    @readonlyWrapper
    def addDataset(self,hdfdir,name,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkGroup(hdfdir)
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

    @readonlyWrapper
    def addDatasets(self,hdfdir,data,append=False,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6,**kwargs):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select HDF5 group
        if hdfdir not in self.fileObj.keys():
            self.mkGroup(hdfdir)
        g = self.fileObj[hdfdir]
        # Write data to group
        for n in data.dtype.names:
            self.add_dataset(hdfdir,n,data[n],append=append,overwrite=overwrite,\
                                 maxshape=maxshape,chunks=chunks,compression=compression,\
                                 compression_opts=compression_opts,**kwargs)
        return

    @readonlyWrapper
    def rmDataset(self,hdfdir,dataset):
        g = self.fileObj[hdfdir]
        if dataset in g.keys():
            del g[dataset]
        return

    def lsDatasets(self,hdfdir):
        objs = self.lsGroups(hdfdir,recursive=False)             
        dsets = []
        def _is_dataset(obj):
            return isinstance(self.fileObj[hdfdir+"/"+obj],h5py.Dataset)        
        return filter(_is_dataset,objs)


    def readDatasets(self,hdfdir,recursive=False,required=None,exit_if_missing=True):
        """
        read_dataset(): Read one or more HDF5 datasets.

        USAGE:   data = HDF5().read_dataset(hdfdir,[recursive],[required],[exist_if_missing])
        
        Inputs:
               hdfdir : Path to dataset or group of datasets to read.
               recursive : If reading HDF5 group, read recursively down subgroups. 
                           (Default = False)
               required : List of names of datasets to read. If required=None, will read
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
                objs = self.lsGroup(hdfdir,recursive=recursive)                     
            else:
                objs = self.lsDatasets(hdfdir)   
            if required is not None:                
                missing = list(set(required).difference(objs))
                if len(missing) > 0:
                    dashed = "-"*10
                    err = dashed+"\n"+funcname+"(): Following keys are missing from "+\
                        hdfdir+":\n     "+"\n     ".join(missing)+"\n"+dashed
                    print(err)
                    raise KeyError(funcname+"(): Some required keys cannot be found in '"+hdfdir+"'!")
                objs = list(set(objs).intersection(required))                
            # ii) Store in dictionary
            def _store_dataset(obj):
                data[str(obj)] = np.array(self.fileObj[hdfdir+"/"+obj])
            map(_store_dataset,objs)
        return data
                            
    
    ##############################################################################
    # ATTRIBUTES
    ##############################################################################
    
    def readAttributes(self,hdfdir,required=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if required is None:        
            return dict(self.fileObj[hdfdir].attrs)
        else:
            good = list(set(required).intersection(self.fileObj[hdfdir].attrs.keys()))
            bad = list(set(required).difference(self.fileObj[hdfdir].attrs.keys()))            
            if self._verbose:
                if len(bad)>0:
                    linereturn = "\n         "
                    print("WARNING! "+funcname+"(): Following attributes not present in '"+hdfdir+\
                              "':"+linereturn+linereturn.join(bad))
            if len(good)==0:
                return {}
            else:
                return {str(g):self.fileObj[hdfdir].attrs[g] for g in good}
    
    @readonlyWrapper
    def addAttributes(self,hdfdir,attributes,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): '"+hdfdir+"' not found in HDF5 file!")                
        attrib = self.fileObj[hdfdir].attrs
        for att in attributes.keys():
            if att in self.fileObj[hdfdir].attrs.keys():
                if self._verbose:
                        print("WARNING! "+funcname+"(): Attribute '"+att+"' already exists!")
                if overwrite:
                    if self._verbose:
                        print("        Overwriting attribute '"+att+"'")                        
                    attrib.create(att,attributes[att],shape=None,dtype=None)
                else:
                    if self._verbose:
                        print("        Ignoring attribute '"+att+"'")                        
            else:
                attrib.create(att,attributes[att],shape=None,dtype=None)
        return

    @readonlyWrapper
    def rmAttributes(self,hdfdir,attributes=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if hdfdir not in self.fileObj:
            raise KeyError(funcname+"(): '"+hdfdir+"' not found in HDF5 file!")                
        attrib = self.fileObj[hdfdir].attrs
        if attributes is None:
            attributes = attrib.keys()
        for att in attributes:
            if att in attrib.keys():
                attrib.__delitem__(att)
        return
