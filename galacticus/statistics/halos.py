#! /usr/bin/env python

import sys
import numpy as np
import warnings
from ..utils.match_searchsorted import match
from .utils import binstats
from scipy.interpolate import interp1d

class HaloOccupationDistribution(object):

    def __init__(self,massBins):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.massBins = massBins
        dm = self.massBins[1] - self.massBins[0]
        self.binCentres = self.massBins[:-1] + dm/2.0
        self.galaxies = np.zeros(len(self.massBins)-1,dtype=float)
        self.galaxies2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.halos = np.zeros(len(self.massBins)-1,dtype=float)
        return

    def reset(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = np.zeros(len(self.massBins)-1,dtype=float)
        self.galaxies2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.halos = np.zeros(len(self.massBins)-1,dtype=float)
        return

    def _addToHOD(self,halos):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        halos = halos.view(np.recarray)
        sumHalos,bins = np.histogram(np.log10(halos.mass),self.massBins)
        self.halos += sumHalos.astype(float)
        sumGalaxies,bin_edges,binnumber = binstats(np.log10(halos.mass),halos.ngals,self.massBins,statistic="sum")
        self.galaxies += sumGalaxies.astype(float)
        sumGalaxies2,bin_edges,binnumber = binstats(np.log10(halos.mass),halos.ngals**2,self.massBins,statistic="sum")
        self.galaxies2 += sumGalaxies2.astype(float)
        return

    def addHalos(self,galaxies,massName="nodeMass200.0",mask=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        galaxies = galaxies.view(np.recarray)
        # Check require properties are present
        required = [massName,"nodeIsIsolated","nodeIndex","parentIndex"]
        if not set(required).issubset(galaxies.dtype.names):
            missing = list(set(required).difference(galaxies.dtype.names))
            raise KeyError(funcname+"(): Following required properties not found: "+",".join(missing)+".")
        # Check mask is provided -- if not then provide simple mask
        if mask is None:
            if len(list(set(["diskMassStellar","spheroidMassStellar"]).intersection(galaxies.dtype.names)))==2:
                totalMass = galaxies.diskMassStellar + galaxies.spheroidMassStellar
                mask = totalMass > 0.0
            else:
                mask = np.ones(len(galaxies.nodeIndex))
        # Find host halos
        hostHalo = galaxies.nodeIsIsolated==1
        totalHostHalos = len(galaxies.nodeIndex[hostHalo])
        if verbose:
            print(funcname+"(): located "+str(totalHostHalos)+" host halos...")
        hostHalos = np.zeros(totalHostHalos,dtype=[("ngals",int),("mass",float),("ID",int)]).view(np.recarray)
        hostHalos.ID = np.copy(galaxies.nodeIndex[hostHalo])
        hostHalos.mass = np.copy(galaxies[massName][hostHalo])
        # Count central galaxies
        if verbose:
            print(funcname+"(): counting central galaxies...")
        galaxyIDs = galaxies.nodeIndex[np.logical_and(mask,galaxies.nodeIsIsolated==1)]
        index = match(galaxyIDs,hostHalos.ID)
        hostHalos.ngals[index] += 1
        # Count satellite galaxies
        if verbose:
            print(funcname+"(): counting satellite galaxies...")
        galaxyIDs = galaxies.parentIndex[np.logical_and(mask,galaxies.nodeIsIsolated==0)]
        index = match(galaxyIDs,hostHalos.ID)
        if any(index==-1):
            problemHalos = index==-1
            percentage = 100.0*float(len(index[problemHalos]))/float(len(index))            
            warnings.warn(funcname+"(): "+str(round(percentage,3))+"% of satellite galaxies do not have a host halo!")
            index = index[np.invert(problemHalos)]
        uniqueIndex,number = np.unique(index,return_counts=True)
        hostHalos.ngals[uniqueIndex] += number
        self._addToHOD(hostHalos)
        return

    def computeHOD(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        np.place(self.halos,self.halos==0.0,1.0)
        dm = self.massBins[1] - self.massBins[0]
        bins = self.massBins[:-1] + dm/2.0
        mean = self.galaxies/self.halos
        sigma = np.sqrt((self.galaxies2/self.halos)-mean**2)
        return bins,mean,sigma


    def smoothHOD(self,mass,massRange=None,inputBins=None,inputHOD=None,**kwargs):
        if inputBins is None or inputHOD is None:            
            bins,hod,err = self.computeHOD()
        else:
            bins = inputBins
            hod = inputHOD
        if massRange is None:
            mask = hod >= 0.0
        else:
            mask = np.logical_and(bins>=massRange[0],bins<=massRange[1])
        bins = bins[mask]
        hod = hod[mask]
        f = interp1d(bins,np.log10(hod),**kwargs)
        return 10.0**f(mass)
