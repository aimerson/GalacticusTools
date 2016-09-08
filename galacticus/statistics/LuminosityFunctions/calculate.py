#! /usr/bin/env python

import sys,fnmatch
import copy
import h5py
import numpy as np
from ...hdf5 import HDF5
from ...io import GalacticusHDF5
from ...Luminosities import ergPerSecond
from ...utils.progress import Progress
from ...cosmology import adjustHubble



class LuminosityFunction(object):
    
    def __init__(self,cosmologyDict,magnitudeBins=None,luminosityBins=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store bins for magnitudes/luminosities
        self.magnitudeBins = magnitudeBins
        if self.magnitudeBins is None:
            self.magnitudeBins = np.arange(-40.0,-5.0,0.2)
        self.luminosityBins = luminosityBins
        if self.luminosityBins is None:
            self.luminosityBins = np.linspace(35.0,44.0,100)            
        # Dictionary for cosmology 
        self.cosmology = cosmologyDict
        # Dictionary for redshifts
        self.outputs = {}
        # Dictionary for luminosity
        self.luminosityFunction = {}
        return


    def _redshiftConsistent(self,z,outName,tolerance=0.01,force=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        consistent = True
        if force:
            return consistent
        if len(self.outputs.keys()) == 0:
            return consistent
        if outName not in self.outputs.keys():
            return consistent
        if z == self.outputs[outName]:
            return consistent
        if (np.fabs(z-self.outputs[outName])/z)>tolerance:
            print("WARNING! "+funcname+"(): Redshifts NOT consistent! (Tolerance = "+\
                      str(tolerance)+")")
            consistent = False
        return consistent
    
    def _cosmologyConsistent(self,cosmology,tolerance=0.01,force=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        consistent = True
        if force:
            return consistent
        if any([(np.fabs(self.cosmology[p]-cosmology[p])/self.cosmology[p])>tolerance \
                    for p in self.cosmology.keys()]):
            print("WARNING! "+funcname+"(): Cosmology NOT consistent! (Tolerance = "+\
                      str(tolerance)+")")
            consistent = False
        return consistent


    def _datasetConsistent(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        consistent = fnmatch.fnmatch(datasetName,"*Luminosity*") or fnmatch.fnmatch(datasetName,"*magitude*")
        if not consistent:
            print("WARNING! "+funcname+"(): Datasset '"+datasetName+"' is NOT a luminosity or a magnitude!")
        return consistent

    def _binsConsistent(self,datasetBins,datasetName,tolerance=0.01,force=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        consistent = True
        if fnmatch.fnmatch(datasetName.lower(),"*luminosity*"):
            bins = self.luminosityBins
        elif fnmatch.fnmatch(datasetName.lower(),"*magnitude*"):
            bins = self.magnitudeBins
        else:
            print "dataset = "+datasetName
            print("WARNING! "+funcname+"(): Dataset NOT consistent with luminosity or magnitude!")
            return False
        if bins is None:
            return True
        if len(bins) != len(datasetBins):
            print("WARNING! "+funcname+"(): Number of bins NOT consistent!")
            return False
        if any((np.fabs(bins-datasetBins)/bins)>tolerance):
            print("WARNING! "+funcname+"(): At least one bin NOT consistent! (Tolerance = "+\
                      str(tolerance)+")")
            return False
        return consistent


    def addDataset(self,outName,redshift,datasetName,datasetBins,datasetLF,cosmology,\
                       force=False,overwrite=False,append=True,\
                       zTolerance=0.01,cosmologyTolerance=0.01,binsTolerance=0.01):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        compute = True
        # Check bins consistent
        if not self._binsConsistent(datasetBins,datasetName,force=force):
            return
        # Check cosmology consistent
        if not self._cosmologyConsistent(cosmology,tolerance=cosmologyTolerance,force=force):
            return
        # Check if dataset exists, whether cosmology/redshifts consistent and whether overwrite
        if outName in self.luminosityFunction.keys():
            if not self._redshiftConsistent(redshift,outName,tolerance=zTolerance,force=force):
                return
            compute = overwrite or append
        else:
            self.outputs[outName] = redshift
            self.luminosityFunction[outName] = {}
            compute = True
        if not compute:
            return
        # Store luminosity function        
        if datasetName in self.luminosityFunction[outName].keys():
            if overwrite:
                self.luminosityFunction[outName][datasetName] = np.copy(datasetLF)
            if append:
                self.luminosityFunction[outName][datasetName] += np.copy(datasetLF)
        else:
            self.luminosityFunction[outName][datasetName] = np.copy(datasetLF)
        return

    
    def addLuminosityFunctions(self,lfObj,force=False,overwrite=False,append=True,\
                                   zTolerance=0.01,cosmologyTolerance=0.01,binsTolerance=0.01,\
                                   verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): adding luminosity functions...")                
        # IF provided with file name, create GalacticusLuminosityFunction object
        if type(lfObj) is str:
            lfObj = GalacticusLuminosityFunction(lfObj)    
        # Check whether bins consistent
        name = "totalLineLuminosity:balmerAlpha65653"
        luminosityConsistent = self._binsConsistent(lfObj.luminosityBins,name,\
                                                        tolerance=binsTolerance,force=force)
        name = "totalMagnitude:SDSS_r"
        magnitudeConsistent = self._binsConsistent(lfObj.magnitudeBins,name,\
                                                       tolerance=binsTolerance,force=force)
        consistent = luminosityConsistent and magnitudeConsistent
        if not consistent:
            return
        # Check whether cosmology consistent
        if not self._cosmologyConsistent(lfObj.cosmology,tolerance=cosmologyTolerance,force=force):
            return
        # Import luminosity functions        
        if len(lfObj.luminosityFunction.keys()) > 0:
            PROG = Progress(len(lfObj.luminosityFunction.keys()))       
            for outKey in lfObj.luminosityFunction.keys():
                z = lfObj.outputs[outKey]
                if self._redshiftConsistent(z,outKey,tolerance=0.01,force=force):
                    # Luminosities
                    luminosities = fnmatch.filter(lfObj.luminosityFunction[outKey].keys(),"*LineLuminosity*")                
                    dummy = [self.addDataset(outKey,z,name,self.luminosityBins,\
                                                 lfObj.luminosityFunction[outKey][name],lfObj.cosmology,\
                                                 force=True,overwrite=overwrite,append=append) for name in luminosities]
                    del dummy
                    # Magnitudes
                    magnitudes = fnmatch.filter(lfObj.luminosityFunction[outKey].keys(),"*Magnitude*")                
                    dummy = [self.addDataset(outKey,z,name,self.magnitudeBins,\
                                                 lfObj.luminosityFunction[outKey][name],lfObj.cosmology,\
                                             force=True,overwrite=overwrite,append=append) for name in magnitudes]
                    del dummy
                PROG.increment()
                if verbose:
                    PROG.print_status_line(task=" adding z = "+str(z))
        else:
            f = HDF5(lfObj.file,'r')
            PROG = Progress(len(f.fileObj["Outputs"].keys()))       
            for outKey in f.fileObj["Outputs"].keys():
                z = float(f.readAttributes("Outputs/"+outKey,required=['redshift'])["redshift"])                
                if self._redshiftConsistent(z,outKey,tolerance=0.01,force=force):
                    keys = f.lsDatasets("Outputs/"+outKey)
                    # Luminosities
                    luminosities = fnmatch.filter(keys,"*LineLuminosity*")
                    dummy = [self.addDataset(outKey,z,name,self.luminosityBins,\
                                                 np.array(f.fileObj["Outputs/"+outKey][name]),lfObj.cosmology,\
                                                 force=True,overwrite=overwrite,append=append) for name in luminosities]
                    del dummy
                    # Magnitudes
                    magnitudes = fnmatch.filter(keys,"*Magnitude*")
                    dummy = [self.addDataset(outKey,z,name,self.magnitudeBins,\
                                                 np.array(f.fileObj["Outputs/"+outKey][name]),lfObj.cosmology,\
                                                 force=True,overwrite=overwrite,append=append) for name in magnitudes]
                    del dummy
                PROG.increment()
                if verbose:
                    PROG.print_status_line(task=" adding z = "+str(z))
            f.close()
        return


    def applyWeight(self,weight,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): Re-weighting luminosity function data using weight = "+str(weight)+" ...")            
        for outKey in self.luminosityFunction.keys():
            for p in self.luminosityFunction[outKey].keys():
                self.luminosityFunction[outKey][p] *= weight
        return        


    def computeDataset(self,outName,redshift,datasetName,datasetValues,cosmology,\
                           weight=None,force=False,overwrite=False,append=True,\
                           zTolerance=0.01,cosmologyTolerance=0.01):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        compute = True
        # Check cosmology consistent
        if not self._cosmologyConsistent(cosmology,tolerance=cosmologyTolerance,force=force):
            return
        # Check if dataset exists, whether cosmology/redshifts consistent and whether overwrite
        if outName in self.outputs.keys():
            if not self._redshiftConsistent(redshift,outName,tolerance=zTolerance,force=force):
                return
            compute = overwrite or append
        else:
            if not self._datasetConsistent(datasetName):
                return
            self.outputs[outName] = redshift
            self.luminosityFunction[outName] = {}
            compute = True
        if not compute:
            return
        # Compute and store luminosity function        
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):            
            values = datasetValues
            bins = self.luminosityBins
        else:
            values = datasetValues
            bins = self.magnitudeBins                
        if weight is None:
            weight = np.ones_like(values)
        lf,bins = np.histogram(values,bins=bins,weights=weight)            
        if datasetName in self.luminosityFunction[outName].keys():
            if overwrite:
                self.luminosityFunction[outName][datasetName] = np.copy(lf)
            if append:
                self.luminosityFunction[outName][datasetName] += np.copy(lf)
        else:
            self.luminosityFunction[outName][datasetName] = np.copy(lf)
        return


    def _writeDatasetToHDF5(self,fileObj,path,datasetName,datasetValue,binWidth=None):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name         
        if len(fnmatch.filter(datasetName.split(":"),"z*"))>0:
            redshiftLabel = fnmatch.filter(datasetName.split(":"),"z*")[0]
            property = datasetName.replace(":"+redshiftLabel,"")
        else:
            property = datasetName
        values = datasetValue    
        if binWidth is not None:
            if binWidth > 0:
                values /= binWidth
        fileObj.addDataset(path,property,np.copy(values),chunks=True,compression="gzip",\
                               compression_opts=6)
        return


    def writeToHDF5(self,hdf5File,verbose=False,divideBinWidth=True,modifyBins=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): Writing luminosity function data to "+hdf5File+" ...")            
        fileObj = HDF5(hdf5File,'w')
        # Write value of cosmological parameters
        fileObj.mkGroup("Cosmology")
        dummy = [fileObj.addAttributes("Cosmology",{key:self.cosmology[key]}) \
                     for key in self.cosmology.keys()]
        # Write luminosity and magnitude bins
        luminosityBinWidth = self.luminosityBins[1] - self.luminosityBins[0]
        if modifyBins:
            luminosityBins = self.luminosityBins[:-1] + luminosityBinWidth/2.0
        else:
            luminosityBins = self.luminosityBins
        fileObj.addDataset("/","luminosityBins",luminosityBins,chunks=True,compression="gzip",\
                               compression_opts=6)
        fileObj.addAttributes("/luminosityBins",{"units":"log10(erg/s)"})
        magnitudeBinWidth = self.magnitudeBins[1] - self.magnitudeBins[0]
        if modifyBins:
            magnitudeBins = self.magnitudeBins[:-1] + magnitudeBinWidth/2.0
        else:
            magnitudeBins = self.magnitudeBins
        fileObj.addDataset("/","magnitudeBins",magnitudeBins,chunks=True,compression="gzip",\
                               compression_opts=6)        
        # Create outputs group and write data for each output in luminosity functions dictionary        
        fileObj.mkGroup("Outputs")
        PROG = Progress(len(self.luminosityFunction.keys()))
        for outstr in self.luminosityFunction.keys():            
            fileObj.mkGroup("Outputs/"+outstr)
            z = self.outputs[outstr]
            fileObj.addAttributes("Outputs/"+outstr,{"redshift":z})
            path = "Outputs/"+outstr+"/"
            # i) Write luminosities
            binWidth = None
            if divideBinWidth:
                binWidth = luminosityBinWidth
            luminosities = fnmatch.filter(self.luminosityFunction[outstr].keys(),"*LineLuminosity*")        
            dummy = [ self._writeDatasetToHDF5(fileObj,path,name,self.luminosityFunction[outstr][name],binWidth=binWidth)\
                          for name in luminosities]
            del dummy
            # ii) Write magnitudes
            binWidth = None
            if divideBinWidth:
                binWidth = magnitudeBinWidth
            magnitudes = fnmatch.filter(self.luminosityFunction[outstr].keys(),"*Magnitude*")
            dummy = [ self._writeDatasetToHDF5(fileObj,path,name,self.luminosityFunction[outstr][name],binWidth=binWidth)\
                          for name in magnitudes]
            del dummy
            PROG.increment()
            PROG.print_status_line(task="z = "+str(z))
        fileObj.close()
        if verbose:
            print(funcname+"(): luminosity function data successfully written to "+hdf5File)
        return


    
class ComputeLuminosityFunction(LuminosityFunction):
    
    def __init__(self,galacticusFile,magnitudeBins=None,luminosityBins=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Open Galacticus file
        self.galacticusFile = galacticusFile
        self.galHDF5Obj = GalacticusHDF5(self.galacticusFile,'r')        
        # Store cosmology
        COS = {}
        COS["HubbleParameter"] = self.galHDF5Obj.parameters["HubbleConstant"]/100.0
        COS["OmegaMatter"] = self.galHDF5Obj.parameters["OmegaMatter"]
        COS["OmegaBaryon"] = self.galHDF5Obj.parameters["OmegaBaryon"]
        COS["OmegaDarkEnergy"] = self.galHDF5Obj.parameters["OmegaDarkEnergy"]
        COS["sigma8"] = self.galHDF5Obj.parameters["sigma_8"]
        COS["ns"] = self.galHDF5Obj.parameters["index"]        
        # Initalise LuminosityFunction class
        super(ComputeLuminosityFunction, self).__init__(COS,magnitudeBins=magnitudeBins,\
                                                            luminosityBins=luminosityBins)        
        return

    def processOutputs(self,redshifts,props=None,incTopHatFilters=False,overwrite=False,verbose=0):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Loop over redshifts
        if verbose > 0:
            print(funcname+"(): Processing redshift outputs...")
        PROG = Progress(len(redshifts))
        for z in redshifts:
            # Locate and select output
            iselect = np.argmin(np.fabs(self.galHDF5Obj.outputs.z-z))
            outstr = "Output"+str(self.galHDF5Obj.outputs["iout"][iselect])
            redshift = self.galHDF5Obj.outputs["z"][iselect]
            if outstr in self.luminosityFunction.keys() and not overwrite:
                if verbose == 2:
                    print(funcname+"(): Luminosity functions for "+outstr+" already computed.")
                continue
            if verbose == 2:
                print(funcname+"(): Processing luminosity functions for "+outstr+" (z = "+str(redshift)+") ...")
            out = self.galHDF5Obj.selectOutput(z)        
            # Calculate weights
            cts = np.array(out["mergerTreeCount"])
            wgt = np.array(out["mergerTreeWeight"])
            weight = np.copy(np.repeat(wgt,cts))            
            # Get properties to process        
            allProps = self.galHDF5Obj.availableDatasets(z)
            if props is None:
                goodProps = fnmatch.filter(allProps,"Magnitude*")
                if not incTopHatFilters:
                    topHats = fnmatch.filter(allProps,"magnitude*emissionLineContinuum*") + \
                        fnmatch.filter(allProps,"Magnitude*topHat*")
                    if len(topHats) > 0:
                        goodProps = list(set(props).difference(topHats))
                goodProps = goodProps + fnmatch.filter(allProps,"*LineLuminosity*")
            else:
                goodProps = []
                for p in props:
                    goodProps = goodProps + fnmatch.filter(allProps,p)  
            if verbose == 2:
                print(funcname+"(): Identified "+str(len(goodProps))+" properties for processing...")
            # Compute and store luminosity functions
            if verbose == 2:
                print(funcname+"(): Computing luminosity functions...")
            # i) Luminosities
            luminosities = fnmatch.filter(goodProps,"*LineLuminosity*")
            dummy = [ self.computeDataset(outstr,z,name,np.log10(ergPerSecond(np.array(out["nodeData/"+name])+1.0e-20)),\
                                              self.cosmology,weight=weight,force=False,overwrite=overwrite,append=True,\
                                              zTolerance=0.01,cosmologyTolerance=0.01) \
                          for name in luminosities ]
            del dummy
            # ii) Magnitudes
            magnitudes = fnmatch.filter(goodProps,"*Magnitude*")
            dummy = [ self.computeDataset(outstr,z,name,np.array(out["nodeData/"+name]),\
                                              self.cosmology,weight=weight,force=False,\
                                              overwrite=overwrite,append=True,\
                                              zTolerance=0.01,cosmologyTolerance=0.01) \
                          for name in magnitudes ]
            del dummy
            PROG.increment()
            if verbose > 0:
                PROG.print_status_line()
        return




class GalacticusLuminosityFunction(LuminosityFunction):
    
    def __init__(self,luminosityFunctionFile):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = luminosityFunctionFile
        f = HDF5(self.file,'r')
        # Read cosmological parameters
        COS = {key:f.readAttributes("Cosmology")[key] for key in f.readAttributes("Cosmology").keys()}
        # Read bins arrays
        bins = f.readDatasets("/",required=["luminosityBins","magnitudeBins"])
        luminosityBins = np.copy(bins["luminosityBins"])
        magnitudeBins = np.copy(bins["magnitudeBins"])
        del bins
        # Initalise LuminosityFunction class
        super(GalacticusLuminosityFunction, self).__init__(COS,magnitudeBins=magnitudeBins,\
                                                               luminosityBins=luminosityBins)
        # Read list of available outputs and datasets
        outputs = list(map(str,f.lsGroups("Outputs")))
        self.availableDatasets = None
        for out in outputs:
            self.outputs[out] = f.readAttributes("Outputs/"+out)["redshift"]        
            outputDatasets = f.lsDatasets("Outputs/"+out)
            if self.availableDatasets is None:
                self.availableDatasets = copy.copy(outputDatasets)
            else:
                self.availableDatasets = list(set(outputDatasets).intersection(self.availableDatasets))            
        self.availableDatasets = list(map(str,self.availableDatasets))
        f.close()        
        return
    
    def getDatasets(self,z,required=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Locate appropriate output by nearest redshift
        redshifts = np.array([self.outputs[key] for key in self.outputs.keys()])        
        iselect = np.argmin(np.fabs(redshifts.astype(float)-float(z)))
        outstr = self.outputs.keys()[iselect]        
        path = "Outputs/"+outstr
        if verbose:
            print_str = funcname+"(): Reading luminosity function(s) for "+outstr+\
                " ( z = "+str(redshifts[iselect])+")"
            print(print_str)
        # Extract properties either from self.luminosityFunction if stored in memory
        # or by reading the luminosity function file
        if outstr in self.luminosityFunction.keys():
            availableDatasets = list(map(str,self.luminosityFunction[outstr].keys()))
            datasets = []
            for req in required:
                datasets = datasets + fnmatch.filter(availableDatasets,req)
            values = {name:self.self.luminosityFunction[outstr][name] for name in datasets}
        else:
            f = HDF5(self.file,'r')
            availableDatasets = list(map(str,f.lsDatasets(path)))
            datasets = []
            for req in required:
                datasets = datasets + fnmatch.filter(availableDatasets,req)
            values = f.readDatasets(path,required=datasets)
            f.close()
        return values

    
