#! /usr/bin/env python


import sys
import fnmatch
import numpy as np
from ...hdf5 import HDF5
from ...io import GalacticusHDF5
from ...Luminosities import ergPerSecond
from ...utils.progress import Progress


class GalacticusLuminosityFunction(object):
    
    def __init__(self,galHDF5Obj,magnitudeBins=None,luminosityBins=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store Galacticus file object
        self.galHDF5Obj = galHDF5Obj        
        self.hubble = self.galHDF5Obj.parameters["HubbleConstant"]/100.0
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

    def processOutput(self,z,props=None,incTopHatFilters=False,overwrite=False,verbose=False):        
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
            goodProps = fnmatch.filter(allProps,"*LuminositiesStellar*")
            if not incTopHatFilters:
                topHats = fnmatch.filter(allProps,"*LuminositiesStellar:emissionLineContinuum*") + \
                    fnmatch.filter(allProps,"*LuminositiesStellar:topHat*")
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
        if verbose:
            print(funcname+"(): Computing luminosity functions...")
        PROG = Progress(len(goodProps))
        for p in goodProps:
            values = np.array(out["nodeData/"+p])
            if fnmatch.fnmatch(p,"*LineLuminosity*"):
                values = np.log10(ergPerSecond(values))
                bins = self.luminosityBins
            else:
                bins = self.magnitudeBins
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
            
        else:
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
                if outKey in addLF.keys():
                    for p in self.luminosityFunction[outKey].keys():
                        if p in addLF[outKey].keys():
                            self.luminosityFunction[outKey][p] += addLF[outKey][p]
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
        fileObj.addAttributes("/",{"hubbleParameter":self.hubble})
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
                lfData = self.luminosityFunction[outstr][p]
                fileObj.addDataset(path,p,lfData,chunks=True,compression="gzip",\
                                       compression_opts=6)                        
        fileObj.close()
        if verbose:
            print(funcname+"(): luminosity function data successfully written to "+hdf5File)
        return








