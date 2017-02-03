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

    def __init__(self,outfile,checkParameters=True,zRange=(None,None)):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(mergeHDF5Outputs, self).__init__(outfile,"a")        
        self._initialCopy = True
        self._checkParameters = checkParameters
        # Store redshift range to merge
        self.zRange = zRange
        return



    def maskRedshifts(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.zRange[0] is not None:
            zLow = z>=self.zRange[0]
        else:
            zLow = np.ones(len(z),dtype=bool)
        if self.zRange[1] is not None:
            zUpp = z<=self.zRange[1]
        else:
            zUpp = np.ones(len(z),dtype=bool)
        return np.logical_and(zLow,zUpp)


    def expansionFactorConsistent(self,a1,a2,tolerance=1.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return np.fabs(a1-a2)<=tolerance

    
    def propertiesConsistent(self,props1,props2,raiseError=True):
        missingProps = list(set(props1).difference(props2))
        if len(missingProps)>0:
            consistent = False
            if raiseError:
                missingStr = "\n    MISSING PROPERTIES:\n    "+"\n    ".join(missingProps)
                raise KeyError(funcname+"(): some datasets are missing!"+str(missingStr))
        else:
            consistent = True
        return consistent


    def selectOutputsToProcess(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        zmask = self.maskRedshifts(galHDF5Obj.outputs.z)
        if not all(zmask):
            minTrue = np.where(zmask)[0].min()
            if minTrue > 0:
                minTrue =- 1
                zmask[minTrue] = True
            maxTrue = np.where(zmask)[0].max()
            if maxTrue < len(zmask)-1:
                maxTrue =+ 1
                zmask[maxTrue] = True        
        return galHDF5Obj.outputs.name[zmask]


    def appendDataset(self,galHDF5Obj,datasetName,sourceOutputName,destOutputName=None,mask=None,append=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set source and destination locations
        sourceDir = "Outputs/"+sourceOutputName+"/nodeData/"
        if destOutputName is None:
            destOutputName = sourceOutputName
        destDir = "Outputs/"+destOutputName+"/nodeData/"
        # Read data and any attributes from source
        if datasetName.lowercase() == "weight":            
            mergerTreeSource = "Outputs/"+sourceOutputName+"/"
            cts = np.array(galHDF5Obj.fileObj[mergerTreeSource+"mergerTreeCount"])
            wgt = np.array(galHDF5Obj.fileObj[mergerTreeSource+"mergerTreeWeight"])
            data = np.copy(np.repeat(wgt,cts))
            del cts,wgt
        else:
            data = np.array(galHDF5Obj.fileObj[sourceDir+datasetName])
        if mask is not None:
            data = data[mask]
        if not append:
            attrib = galHDF5Obj.readAttrbiutes(sourceDir+datasetName)            
        # Write to destination
        self.addDataset(destDir,datasetName,data,append=append)
        if not append:
            self.addAttributes(destDir+datasetName,attrib,overwrite=False)
        return


    def appendOutput(self,galHDF5Obj,outName,expansionFactorTolerance=1.0e-6):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get redshift for this output
        iz = np.where(galHDF5Obj.outputs.name==outName)[0][0]
        zOutput = galHDF5Obj.outputs.z[iz]
        # Check this output contains galaxies
        outputProps = galHDF5Obj.availableDatasets(zOutput)
        if len(outputProps)==0:
            # No galaxies!
            return
        # Add 'galaxy weight' proeprty
        outputProps.append("weight")
        # Check expansion factors are consistent
        aSource = galHDF5Obj.outputs.a[iz]        
        aDestination = float(self.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"])
        if not self.expansionFactorConsistent(aSource,aDestination,tolerance=expansionFactorTolerance):            
            raise ValueError(funcname+"(): cannot append outputs -- expansion factors not consistent within tolerance!")
        # Check output within redshift range (if specified) or return mask if lightcone redshifts available
        if "lightconeRedshift" not in outputProps:            
            if self.maskRedshifts(zOutput):
                zmask = np.ones(len(galHDF5Obj.fileObj["Outputs/"+outName+"/"+outputProps[0]]),dtype=bool)
            else:
                # Not a lightcone and snapshot not in redshift range
                return
        else:
            # Is a lightcone so want to copy only galaxies inside redshift range
            zmask = self.maskRedshifts(np.array(galHDF5Obj.fileObj["Outputs/"+outName+"/lightconeRedshift"]))
        # Check properties are consistent
        destinationProps = list(map(str,self.fileObj["Outputs/"+outName+"/nodeData"].keys()))        
        if not len(destinationProps)==0:
            if not self.propertiesConsistent(outputProps,destinationProps):
                raise KeyError(funcname+"(): datasets not consistent (some datasets missing)!")
        if len(destinationProps)==0:
            append = False
        else:
            append = True
        # Begin copying datasets
        dummy = [self.appendDataset(galHDF5Obj,datasetName,outName,destOutputName=outName,\
                                        mask=zmask,append=append) for datasetName in outputProps]
        return


    def buildOutputsDirectory(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Make group to store outputs
        self.mkGroup("Outputs")
        outputNames = self.selectOutputsToProcess(galHDF5Obj)
        for outputName in outputNames:
            self.mkGroup("Outputs/"+outputName)
            attrib = galHDF5Obj.readAttributes("Outputs/"+outputName)
            self.addAttributes("Outputs/"+outputName,attrib)
        return



    def appendFile(self,ifile,progressOBJ=None,ignoreParameters=[]):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        GAL = GalacticusHDF5(ifile,"r")
        if GAL.outputs is None:
            if progressOBJ is not None:
                progressOBJ.increment()
                progressOBJ.print_status_line()
            return
        if self._initialCopy:
            # Copy all directories apart from galaxy outputs
            cpDirs = list(set(GAL.fileObj.keys()).difference(["Outputs"]))            
            dummy = [self.cpGroup(ifile,group) for group in cpDirs]
            # Build directory structure to store galaxy data
            self.buildOutputsDirectory(GAL)
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
        # Get list of output names to process
        outputNames = self.fileObj["Outputs"].keys()
        # Check parameters are consistent
        if self._checkParameters:
            if not compareParameterSets(GAL.parameters,self.parameters,ignore=ignoreParameters):
                self.fileObj.close()
                raise ValueError(funcname+"(): cannot add file -- parameters not consistent!")        
        # Append galaxy data
        dummy = [self.appendOutput(GAL,outName) for outName in GAL.fileObj["Outputs"].keys()]
        if progressOBJ is not None:
            progressOBJ.increment()
            progressOBJ.print_status_line()
        return





    # Depracated functions -- to be removed after testing...

    def addOutput(self,GAL,outName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get redshift for this output
        aGAL = float(GAL.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"])
        zGAL = (1.0/aGAL)-1.0
        # Check this output contains any galaxy information
        propsGAL = GAL.availableDatasets(zGAL)
        if len(propsGAL) == 0:
            return
        # Check this output is inside the desired redshift range
        aSelf = self.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"]
        aGAL = GAL.fileObj["Outputs/"+outName].attrs["outputExpansionFactor"]
        if aSelf != aGAL:
            raise ValueError(funcname+"(): cannot append outputs -- expansion factors do not match!")
        propsSelf = list(map(str,self.fileObj["Outputs/"+outName+"/nodeData"].keys()))        
        propsGAL = list(map(str,GAL.fileObj["Outputs/"+outName+"/nodeData"].keys()))
        if len(propsGAL) == 0:
            return
        missingProps = list(set(propsSelf).difference(propsGAL))
        if len(missingProps)>0:
            self.fileObj.close()
            missingStr = "\n    MISSING PROPERTIES:\n    "+"\n    ".join(missingProps)
            raise KeyError(funcname+"(): cannot append output -- some datasets are missing!"+str(missingStr))
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
        dummy = [self.addOutput(GAL,outName) for outName in GAL.fileObj["Outputs"].keys()]
        if progressOBJ is not None:
            progressOBJ.increment()
            progressOBJ.print_status_line()
        return
