#! /usr/bin/env python

import sys,os,glob,fnmatch
import numpy as np
from ..io import GalacticusHDF5
from ..hdf5 import HDF5
from ..parameters import compareParameterSets


def copyHDF5File(ifile,ofile,overwrite=False):
    if not os.path.exists(ifile):
        raise IOError("compyHDF5File(): input file '"+ifile+"' does NOT exist!")
    if os.path.exists(ofile) and not overwrite:
        return
    OUT = HDF5(ofile,'w')
    GAL = GalacticusHDF5(ifile,'r')
    dummy = [OUT.cpGroup(ifile,group) for group in GAL.fileObj.keys()]
    return


class mergeHDF5Outputs(HDF5):

    def __init__(self,outfile,checkParameters=True):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(mergeHDF5Outputs, self).__init__(outfile,"a")        
        self._initialCopy = True
        self._checkParameters = checkParameters
        return

    def addOutput(self,GAL,outName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        aSelf = self.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"]
        aGAL = GAL.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"]
        if not fnmatch(aSelf,aGAL):
            raise ValueError(funcname+"(): cannot append outputs -- expansion factors do not match!")
        propsSelf = self.fileObj["Outputs/"+outName+"/nodeData"].keys()
        propsGAL = GAL.fileObj["Outputs/"+outName+"/nodeData"].keys()
        missingProps = list(set(propsSelf).difference(propsGAL))
        if len(missingProps)>0:
            self.fileObj.close()
            missingStr = "\n    MISSING PROPERTIES:\n    "+"\n    ".join(missingProps)
            raise KeyError(funcname+"(): cannot append output -- some datasets are missing!"+missingStr)
        dummy = [self.addDataset("Outputs/"+outName+"/nodeData/",datasetName,np.array(GAL.fileObj["Outputs/"+outName+"/nodeData/"+datasetName]),append=True)\
                     for datasetName in propsGAL]
        return

    def addFile(self,ifile,progressOBJ=None,ignoreParameters=[]):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        GAL = GalacticusHDF5(ifile,"r")
        if GAL.outputs is None:
            if progressOBJ is not None:
                progressOBJ.increment()
                progressOBJ.print_status_line()
            return
        if self._initialCopy:
            dummy = [self.cpGroup(ifile,group) for group in GAL.fileObj.keys()]
            # Store parameters
            self.parameters = dict(GAL.fileObj["Parameters"].attrs)
            self.parameters_parents = { k:"parameters" for k in GAL.fileObj["Parameters"].attrs.keys()}
            for k in GAL.fileObj["Parameters"]:
                if len(GAL.fileObj["Parameters/"+k].attrs.keys())>0:
                    d = dict(GAL.fileObj["Parameters/"+k].attrs)
                    self.parameters.update(d)
                    d = { a:k for a in GAL.fileObj["Parameters/"+k].attrs.keys()}
                    self.parameters_parents.update(d)            
            self._initialCopy = False
        else:
            outputNames = self.fileObj["Outputs"].keys()
        if self._checkParameters:
            if not compareParameterSets(GAL.parameters,self.parameters,ignore=ignoreParameters):
                self.fileObj.close()
                raise ValueError(funcname+"(): cannot add file -- parameters not consistent!")        
        dummy = [self.addOutput(self,GAL,outName) for outName in GAL["Outputs"].keys()]
        if progressOBJ is not None:
            progressOBJ.increment()
            progressOBJ.print_status_line()
        return
