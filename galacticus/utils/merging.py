#! /usr/bin/env python

import sys,os,glob,fnmatch
import numpy as np
from ..io import GalacticusHDF5
from ..hdf5 import HDF5
from copy import copy
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

    def __init__(self,outfile,zMin=None,zMax=None,expansionFactorTolerance=1.0e-6,checkParameters=True):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(mergeHDF5Outputs, self).__init__(outfile,"w")        
        # Store redshift range to merge
        self.zMin = zMin
        self.zMax = zMax
        # Set variables and tolerances for consistency checking
        self.checkParameters = checkParameters
        self.expansionFactorTolerance = expansionFactorTolerance
        # Variable to store parameter information
        self.parameters = None
        return
    
    def updateUUID(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        uuid = self.readAttributes("/",required=["UUID"])
        if "UUID" in uuid.keys():
            uuid = uuid["UUID"] + ":"
        else:
            uuid = ""
        uuid = uuid + galHDF5Obj.readAttributes("/",required=["UUID"])["UUID"]
        self.addAttributes("/",{"UUID":uuid},overwrite=True)
        return

    def addVersionInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if version information already written
        if "Version" in self.fileObj.keys():
            return 
        # Copy version information
        self.cpGroup(galHDF5Obj.filename,"Version")
        return

    def addBuildInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if build information already written
        if "Build" in self.fileObj.keys():
            return 
        # Write build information
        self.cpGroup(galHDF5Obj.filename,"Build")
        return
    

    def addParametersInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.cpGroup(galHDF5Obj.filename,"Parameters")
        attrib = {"treeEvolveWorkerNumber":1,"treeEvolveWorkerCount":1}
        self.addAttributes("Parameters",attrib,overwrite=True)
        self.parameters = copy(galHDF5Obj.parameters)
        return 
    
    def checkParametersConsistent(self,galHDF5Obj,ignoreParameters=[]):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if parameters information written -- if not write and exit
        if "Parameters" not in list(map(str,self.fileObj.keys())):
            self.addParametersInformation(galHDF5Obj)
            return 
        # Parameters already written -- check parameters are consistent
        if self.checkParameters:
            if not compareParameterSets(galHDF5Obj.parameters,self.parameters,ignore=ignoreParameters):            
                self.fileObj.close()
                err = funcname+"(): cannot merge files -- parameter sets not consistent!" + \
                    "\n"+" "*len(funcname)+"   Input file: "+galHDF5Obj.filename            
                raise ValueError(err)
        return
    
    
    def updateGlobalHistoryDataset(self,galHDF5Obj,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check global history is present
        if "globalHistory" not in self.fileObj.keys():
            self.updateGlobalHistory(galHDF5Obj)
            return
        if datasetName not in ["historyTime","historyExpansion"]:
            newData = np.array(galHDF5Obj.fileObj["globalHistory/"+datasetName])
            data = np.array(self.fileObj["globalHistory/"+datasetName])
            data += newData
            self.addDataset("globalHistory",datasetName,data,overwrite=True)
        return
            
    def updateGlobalHistory(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if any global history data has been written
        if "globalHistory" not in self.fileObj.keys():
            self.cpGroup(galHDF5Obj.filename,"globalHistory")
            return
        # Update global history
        dummy = [self.updateGlobalHistoryDataset(galHDF5Obj,name) \
                     for name in self.fileObj["globalHistory"].keys()]
        del dummy
        return
    

    def maskRedshifts(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        zLow = np.ones(len(z),dtype=bool)
        zUpp = np.ones(len(z),dtype=bool)
        if self.zMin is not None:
            zLow = z>=self.zMin
        if self.zMax is not None:
            zUpp = z<=self.zMax
        return np.logical_and(zLow,zUpp)


    def getOutputsToMerge(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print self.zMin,self.zMax
        print galHDF5Obj.outputs.z
        # If no redshift limits specified simply return list of outputs
        if self.zMin is None and self.zMax is None:
            return galHDF5Obj.outputs.name
        # Check if outputs contain lightcone redshifts
        lightcone = "lightconeRedshift" in galHDF5Obj.fileObj["Outputs/Output1"].keys()
        # Apply redshift mask
        zmask = self.maskRedshifts(galHDF5Obj.outputs.z)
        if lightcone and not all(zmask):
            minTrue = np.where(zmask)[0].min()
            if minTrue > 0:
                minTrue =- 1
                zmask[minTrue] = True
            maxTrue = np.where(zmask)[0].max()
            if maxTrue < len(zmask)-1:
                maxTrue =+ 1
                zmask[maxTrue] = True
        print galHDF5Obj.outputs.name[zmask]
        return galHDF5Obj.outputs.name[zmask]
        

    def expansionFactorConsistent(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        a = float(self.readAttributes("Outputs/"+outputName,required=["outputExpansionFactor"])["outputExpansionFactor"])
        aNew = float(galHDF5Obj.readAttributes("Outputs/"+outputName,required=["outputExpansionFactor"])["outputExpansionFactor"])
        if not np.fabs(a-aNew)<=self.expansionFactorTolerance:
            self.fileObj.close()
            err = funcname+"(): cannot merge output "+outputname+" -- expansion factors not consistent!" + \
                "\n"+" "*len(funcname)+"   Input file: "+galHDF5Obj.filename
            raise ValueError(err)
        return 
    

    def checkForMissingDatasets(self,datasets,newDatasets,fileName=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        missingDatasets = list(set(datasets).difference(newDatasets))
        if len(missingDatasets)>0:
            err = funcname+"(): some datasets are missing!"
            if fileName is not None:
                err = err + "\n"+" "*len(funcname)+"   Input file: "+fileName
            err = err + "\n\n    MISSING DATASETS:\n    "+"\n    ".join(missingProps)
            raise KeyError(err)
        return

    def mergerTreeDatasetsConsistent(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        mergerTreeDatasets = list(set(self.fileObj["Outputs/"+outputName].keys()).difference(["nodeData"]))
        newMergerTreeDatasets = list(set(galHDF5Obj.fileObj["Outputs/"+outputName].keys()).difference(["nodeData"]))
        self.checkForMissingDatasets(mergerTreeDatasets,newMergerTreeDatasets,fileName=galHDF5Obj.filename)
        return

    def nodeDataDatasetsConsistent(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nodeDataDatasets = self.fileObj["Outputs/"+outputName+"/nodeData"].keys()
        newNodeDataDatasets = galHDF5Obj.fileObj["Outputs/"+outputName+"/nodeData"].keys()
        self.checkForMissingDatasets(nodeDataDatasets,newNodeDataDatasets,fileName=galHDF5Obj.filename)
        return
    
    def outputDatasetsConsistent(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check merger tree datasets consistent
        self.mergerTreeDatasetsConsistent(galHDF5Obj,outputName)
        # Check nodeData datasets are consistent
        self.nodeDataDatasetsConsistent(galHDF5Obj,outputName)
        # No errors! All datasets consistent
        return

    
    def appendDataset(self,galHDF5Obj,hdf5Dir,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        data = self.readDatasets(hdf5Dir,recursive=False,required=[datasetName])
        self.addDataset(hdf5Dir,datasetName,np.array(data[datasetName]),append=True)
        return


    def mergeMergerTreeDatasets(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        mergerTreeDatasets = list(set(self.fileObj["Outputs/"+outputName].keys()).difference(["nodeData"]))
        hdf5Dir = "Outputs/"+outputName+"/"
        dummy = [self.appendDataset(galHDF5Obj,hdf5Dir,name) for name in mergerTreeDatasets]
        del dummy
        return

    def mergeNodeDataDatasets(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nodeDataDatasets = self.fileObj["Outputs/"+outputName+"/nodeData"].keys()
        hdf5Dir = "Outputs/"+outputName+"/nodeData/"
        dummy = [self.appendDataset(galHDF5Obj,hdf5Dir,name) for name in nodeDataDatasets]
        del dummy
        return

    def mergeOutput(self,galHDF5Obj,outputName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print galHDF5Obj.filename,outputName
        # Check Outputs group exists
        if "Outputs" not in self.fileObj.keys():
            self.mkGroup("Outputs")
        # If specified output does not exist, simply copy and exit
        if outputName not in self.fileObj["Outputs/"].keys():
            self.cpGroup(galHDF5Obj.filename,"/Outputs/"+outputName+"/")
            return
        # Check expansion factors of outputs are consistent
        self.expansionFactorConsistent(galHDF5Obj,outputName)
        # Check datasets in this output are consistent
        self.outputDatasetsConsistent(galHDF5Obj,outputName)
        # Merge merger tree datasets
        self.mergeMergerTreeDatasets(galHDF5Obj,outputName)
        # Merge nodeData datasets
        self.mergeNodeDataDatasets(galHDF5Obj,outputName)
        # Finished merging this output!
        return


    def mergeFile(self,ifile,progressOBJ=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Open file
        galHDF5Obj = GalacticusHDF5(ifile,'r')
        # Check files are consistent:
        self.checkParametersConsistent(galHDF5Obj)
        # Update information
        self.updateUUID(galHDF5Obj)
        self.addVersionInformation(galHDF5Obj)
        self.addBuildInformation(galHDF5Obj)
        # Update global history information
        self.updateGlobalHistory(galHDF5Obj)
        # Get outputs to merge
        outputsToMerge = self.getOutputsToMerge(galHDF5Obj)
        # Merge outputs
        dummy = [self.mergeOutput(galHDF5Obj,outputName) \
                     for outputName in outputsToMerge]        
        del dummy
        # Update progress and return
        if progressOBJ is not None:
            progressOBJ.increment()
            progressOBJ.print_status_line(task=galHDF5Obj.filename)
        galHDF5Obj.close()
        return

    
            

