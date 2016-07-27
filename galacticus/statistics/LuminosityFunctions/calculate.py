#! /usr/bin/env python


import fnmatch
import numpy as np
from .io import GalacticusHDF5
from .Luminosities import ergPerSecond
from .utils.progress import Progress


class GalacticusLuminosityFunction(object):
    
    def __init__(self,galHDF5Obj,magnitudeBins=None,luminosityBins=None):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Store Galacticus file object
        self.galHDF5Obj = galHDF5Obj        
        # Store bins for magnitudes/luminosities
        self.magnitudeBins = magnitudeBins
        if self.magnitudeBins is None:
            self.magnitudeBins = np.arange(-40.0,-5.0,0.2)
        self.luminosityBins = luminosityBins
        if self.luminosityBins is None:
            self.luminosityBins = np.linspace(38.0,44.0,100)
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
        weight = np.array(out["nodeData/weight"])
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


        



        








