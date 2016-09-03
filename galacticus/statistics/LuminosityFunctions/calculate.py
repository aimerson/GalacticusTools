#! /usr/bin/env python


import sys,fnmatch
import h5py
import numpy as np
from ...hdf5 import HDF5
from ...io import GalacticusHDF5
from ...Luminosities import ergPerSecond
from ...utils.progress import Progress
from ...cosmology import adjustHubble



class ComputeLuminosityFunction(object):
    
    def __init__(self,galHDF5Obj,hubble=None,magnitudeBins=None,luminosityBins=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store Galacticus file object
        self.galHDF5Obj = galHDF5Obj        
        self.hubbleGalacticus = self.galHDF5Obj.parameters["HubbleConstant"]/100.0
        # Set Hubble parameter to correct to
        if hubble is None:
            self.hubble = self.hubbleGalacticus
        else:
            self.hubble = hubble
        # Store bins for magnitudes/luminosities
        self.magnitudeBins = magnitudeBins
        if self.magnitudeBins is None:
            self.magnitudeBins = np.arange(-40.0,-5.0,0.2)
        self.luminosityBins = luminosityBins
        if self.luminosityBins is None:
            self.luminosityBins = np.linspace(35.0,44.0,100)
        # Dictionary to store results
        self.luminosityFunction = {}
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

    
    def applyWeight(self,weight,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name            
        if verbose:
            print(funcname+"(): Re-weighting luminosity function data using weight = "+str(weight)+" ...")            
        for outKey in self.luminosityFunction.keys():
            for p in self.luminosityFunction[outKey].keys():
                self.luminosityFunction[outKey][p] *= weight
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
                redshiftLabel = fnmatch.filter(p.split(":"),"z*")[0]
                lfData = self.luminosityFunction[outstr][p]                
                if fnmatch.fnmatch(p,"*LineLuminosity*"):
                    lfData /= luminosityBinWidth
                else:
                    lfData /= magnitudeBinWidth
                fileObj.addDataset(path,p.replace(":"+redshiftLabel,""),lfData,chunks=True,compression="gzip",\
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

    
