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
        if any([(np.fabs(self.cosmolgy[p]-cosmology[p])/self.cosmology[p])>tolerance \
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
        if fnmatch.fnmatch(datasetName,"*Luminosity*"):
            bins = self.luminosityBins
        elif fnmatch.fnmatch(datasetName,"*magitude*"):
            bins = self.magnitudeBins
        else:
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
        if outName in self.outputs.keys():
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
        # Check whether bins consistent
        name = "totalLineLuminosity:balmerAlpha65653"
        luminosityConsistent = self._binsConsistent(lfObj.luminosityBins,name,\
                                                        tolerance=binsTolerance,force=force)
        name = "totalMagnitude:SDSS_r"
        magnitudeConsistent = self._binsConsistent(lfObj.luminosityBins,name,\
                                                       tolerance=binsTolerance,force=force)
        consistent = luminosityConsistent and magnitudeConsistent
        if not consistent:
            return
        # Check whether cosmology consistent
        if not self._cosmologyConsistent(lfObj.cosmology,tolerance=cosmologyTolerance,force=force):
            return
        # Import luminosity functions
        PROG = Progress(len(self.luminosityFunction.keys()))       
        for outKey in lfObj.luminosityFunction.keys():
            z = lfObj.outputs[outKey]
            if self._redshiftConsistent(z,outKey,zTolerance=0.01,force=force):
                # Luminosities
                luminosities = fnmatch.filter(lfObj.luminosityFunction[outkey].keys(),"*LineLuminosity*")                
                dummy = [self.addDataset(outKey,z,name,self.luminosityBins,\
                                             lfObj.luminosityFunction[outkey][name],lfObj.cosmology,\
                                             force=True,overwrite=overwrite,append=append) for name in luminosities]
                del dummy
                # Magnitudes
                magnitudes = fnmatch.filter(lfObj.luminosityFunction[outkey].keys(),"*Magnitude*")                
                dummy = [self.addDataset(outKey,z,name,self.magnitudeBins,\
                                             lfObj.luminosityFunction[outkey][name],lfObj.cosmology,\
                                             force=True,overwrite=overwrite,append=append) for name in magnitudes]
                del dummy
            PROG.increment()
            if verbose:
                PROG.print_status_line(task=" adding z = "+str(z))
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
        if fnmatch.fnmatch(p,"*LineLuminosity*"):
            values = np.log10(ergPerSecond(datasetValues))
            bins = self.luminosityBins
        else:
            values = datasetValues
            bins = self.magnitudeBins                
        lf,bins = np.histogram(values,bins=bins,weights=weight)            
        if datasetName in self.luminosityFunction[outName].keys():
            if overwrite:
                self.luminosityFunction[outName][datasetName] = np.copy(lf)
            if append:
                self.luminosityFunction[outName][datasetName] += np.copy(lf)
        else:
            self.luminosityFunction[outName][datasetName] = np.copy(lf)
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

    def processOutput(self,z,props=None,incTopHatFilters=False,addWeight=1.0,overwrite=False,verbose=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Locate output
        iselect = np.argmin(np.fabs(self.galHDF5Obj.outputs.z-z))
        outstr = "Output"+str(self.galHDF5Obj.outputs["iout"][iselect])
        if outstr in self.luminosityFunction.keys() and not overwrite:
            if verbose:
                print(funcname+"(): Luminosity functions for "+outstr+" already computed.")
            return
        if verbose:
            print(funcname+"(): Processing luminosity functions for "+outstr+"...")
        out = self.galHDF5Obj.selectOutput(z)        
        # Get properties to process        
        allProps = self.galHDF5Obj.availableDatasets(z)
        if props is None:
            goodProps = fnmatch.filter(allProps,"magnitude*")
            if not incTopHatFilters:
                topHats = fnmatch.filter(allProps,"magnitude*emissionLineContinuum*") + \
                    fnmatch.filter(allProps,"magnitude*topHat*")
                if len(topHats) > 0:
                    goodProps = list(set(props).difference(topHats))
            goodProps = goodProps + fnmatch.filter(allProps,"*LineLuminosity*")
        else:
            goodProps = []
            for p in props:
                goodProps = goodProps + fnmatch.filter(allProps,p)  
        if verbose:
            print(funcname+"(): Identified "+str(len(goodProps))+" properties for processing...")
        # Compute and store luminosity functions
        zLF = {}
        cts = np.array(out["mergerTreeCount"])
        wgt = np.array(out["mergerTreeWeight"])
        weight = np.copy(np.repeat(wgt,cts))
        weight *= addWeight
        weight = adjustHubble(weight,self.hubbleGalacticus,self.hubble,'density')
        if verbose:
            print(funcname+"(): Computing luminosity functions...")
        PROG = Progress(len(goodProps))
        for p in goodProps:
            values = np.array(out["nodeData/"+p])
            if fnmatch.fnmatch(p,"*LineLuminosity*"):
                values = adjustHubble(values,self.hubbleGalacticus,self.hubble,"luminosity")
                values = np.log10(ergPerSecond(values))
                bins = self.luminosityBins
            else:
                bins = self.magnitudeBins
                values = adjustHubble(values,self.hubbleGalacticus,self.hubble,"magnitude")
            zLF[p],bins = np.histogram(values,bins=bins,weights=weight)
            PROG.increment()
            if verbose:
                PROG.print_status_line()
        self.luminosityFunction[outstr] = zLF            
        return

        
    def addLuminosityFunctions(self,lfClass,binsTolerance=1.0e-3,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): adding luminosity functions...")                
        # Check whether luminosity functions for this class have already been calculated.
        # If not simply store second object as luminosity functions for this class.
        if len(self.luminosityFunction.keys()) == 0:
            self.luminosityFunction = lfClass.luminosityFunction.copy()
            self.magnitudeBins = np.copy(lfClass.magnitudeBins)
            self.luminosityBins = np.copy(lfClass.luminosityBins)
            self.hubble = lfClass.hubble
        else:
            # Check values for Hubble parameter are consistent
            if self.hubble != lfClass.hubble:
                raise ValueError(funcname+"(): Values for Hubble parameter are not consistent!")
            # Check luminosity and magnitude bins for two LF objects are consistent
            binsDiff = np.fabs(self.luminosityBins-lfClass.luminosityBins)
            if any(binsDiff>binsTolerance):
                raise ValueError(funcname+"(): cannot add luminosity functions -- luminosity bins are not consistent!")
            binsDiff = np.fabs(self.magnitudeBins-lfClass.magnitudeBins)
            if any(binsDiff>binsTolerance):
                raise ValueError(funcname+"(): cannot add luminosity functions -- magnitude bins are not consistent!")
            # Add luminosity functions 
            PROG = Progress(len(self.luminosityFunction.keys()))       
            for outKey in self.luminosityFunction.keys():
                if outKey in lfClass.luminosityFunction.keys():
                    for p in self.luminosityFunction[outKey].keys():
                        if p in lfClass.luminosityFunction[outKey].keys():                            
                            self.luminosityFunction[outKey][p] += lfClass.luminosityFunction[outKey][p]
                PROG.increment()
                if verbose:
                    PROG.print_status_line()
        return

    


                    
    def writeToHDF5(self,hdf5File,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): Writing luminosity function data to "+hdf5File+" ...")            
        fileObj = HDF5(hdf5File,'w')
        # Write value of Hubble parameter
        fileObj.mkGroup("Cosmology")
        fileObj.addAttributes("Cosmology",{"HubbleParameter":self.hubble})
        fileObj.addAttributes("Cosmology",{"OmegaMatter":self.galHDF5Obj.parameters["OmegaMatter"]})
        fileObj.addAttributes("Cosmology",{"OmegaBaryon":self.galHDF5Obj.parameters["OmegaBaryon"]})
        fileObj.addAttributes("Cosmology",{"OmegaDarkEnergy":self.galHDF5Obj.parameters["OmegaDarkEnergy"]})
        fileObj.addAttributes("Cosmology",{"sigma8":self.galHDF5Obj.parameters["sigma_8"]})            
        fileObj.addAttributes("Cosmology",{"ns":self.galHDF5Obj.parameters["index"]})
        # Write luminosity and magnitude bins
        luminosityBinWidth = self.luminosityBins[1] - self.luminosityBins[0]
        luminosityBins = self.luminosityBins[:-1] + luminosityBinWidth/2.0
        fileObj.addDataset("/","luminosityBins",luminosityBins,chunks=True,compression="gzip",\
                               compression_opts=6)
        fileObj.addAttributes("/luminosityBins",{"units":"log10(erg/s)"})
        magnitudeBinWidth = self.magnitudeBins[1] - self.magnitudeBins[0]
        magnitudeBins = self.magnitudeBins[:-1] + magnitudeBinWidth/2.0
        fileObj.addDataset("/","magnitudeBins",magnitudeBins,chunks=True,compression="gzip",\
                               compression_opts=6)        
        # Create outputs group and write data for each output in luminosity functions dictionary
        fileObj.mkGroup("Outputs")
        for outstr in self.luminosityFunction.keys():            
            fileObj.mkGroup("Outputs/"+outstr)
            iout = int(outstr.replace("Output",""))
            iselect = np.argwhere(self.galHDF5Obj.outputs.iout==iout)
            z = str(self.galHDF5Obj.outputs["z"][iselect][0][0])            
            fileObj.addAttributes("Outputs/"+outstr,{"redshift":z})
            for p in self.luminosityFunction[outstr].keys():
                path = "Outputs/"+outstr+"/"
                if len(fnmatch.filter(p.split(":"),"z*"))>0:
                    redshiftLabel = fnmatch.filter(p.split(":"),"z*")[0]
                else:
                    redshiftLabel = None
                lfData = self.luminosityFunction[outstr][p]                
                if fnmatch.fnmatch(p,"*LineLuminosity*"):
                    lfData /= luminosityBinWidth
                else:
                    lfData /= magnitudeBinWidth
                property = p
                if redshiftLabel is not None:
                    property = property.replace(":"+redshiftLabel,"")
                fileObj.addDataset(path,property,lfData,chunks=True,compression="gzip",\
                                       compression_opts=6)                        
        fileObj.close()
        if verbose:
            print(funcname+"(): luminosity function data successfully written to "+hdf5File)
        return



class GalacticusLuminosityFunction(object):
    
    def __init__(self,luminosityFunctionFile):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.file = luminosityFunctionFile
        f = HDF5(self.file,'r')
        # Read cosmological parameters
        self.hubble = f.readAttributes("Cosmology")["HubbleParameter"]        
        self.omega0 = f.readAttributes("Cosmology")["OmegaMatter"]        
        self.lambda0 = f.readAttributes("Cosmology")["OmegaDarkEnergy"]        
        self.omegab = f.readAttributes("Cosmology")["OmegaBaryon"]        
        self.sigma8 = f.readAttributes("Cosmology")["sigma8"]        
        self.ns = f.readAttributes("Cosmology")["ns"]                
        # Read bins arrays
        bins = f.readDatasets("/",required=["luminosityBins","magnitudeBins"])
        self.luminosityBins = np.copy(bins["luminosityBins"])
        self.magnitudeBins = np.copy(bins["magnitudeBins"])
        del bins
        # Read list of available outputs and datasets
        self.outputs = list(map(str,f.lsGroups("Outputs")))
        self.redshifts = np.ones(len(self.outputs))
        self.datasets = {}
        for i,out in enumerate(self.outputs):
            self.redshifts[i] = f.readAttributes("Outputs/"+out)["redshift"]
            self.datasets[out] = list(map(str,f.lsDatasets("Outputs/"+out)))
        f.close()
        return

    def constructDictionary(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.luminosityFunction = {}
        f = HDF5(self.file,'r')
        for output in self.outputs:
            outputDict = f.readDatasets("Outputs/"+output,required=self.datasets[output])
            self.luminosityFunction[output] = copy.copy(outputDict)
        f.close()
        return
    
    def getDatasets(self,z,required=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        iselect = np.argmin(np.fabs(self.redshifts-z))
        path = "Outputs/"+self.outputs[iselect]
        if verbose:
            print_str = funcname+"(): Reading luminosity function(s) for z = "+\
                str(self.redshifts[iselect])+"\n         -- located in path "+path
            print(print_str)
        f = HDF5(self.file,'r')
        availableDatasets = list(map(str,f.lsDatasets(path)))
        datasets = []
        for req in required:
            datasets = datasets + fnmatch.filter(availableDatasets,req)
        lfData = f.readDatasets(path,required=datasets)
        f.close()
        return lfData

    
