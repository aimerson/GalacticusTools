#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import warnings
from ..utils.match_searchsorted import match
from .utils import binstats
from ..satellites import getHostIndex
from scipy.interpolate import interp1d

class HaloOccupationDistribution(object):

    def __init__(self,massBins):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set mass bins
        self.massBins = massBins
        dm = self.massBins[1] - self.massBins[0]
        self.binCentres = self.massBins[:-1] + dm/2.0
        # Variables to store galaxy data
        self.MASS = None
        self.CENTRALS = None
        self.SATELLITES = None
        self.WEIGHTS = None
        # Create vectors to store HODs
        self.galaxies = np.zeros(len(self.massBins)-1,dtype=float)
        self.galaxies2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.centrals = np.zeros(len(self.massBins)-1,dtype=float)
        self.centrals2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.satellites = np.zeros(len(self.massBins)-1,dtype=float)
        self.satellites2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.halos = np.zeros(len(self.massBins)-1,dtype=float)
        return

    
    def reset(self):
        self.resetHOD()
        self.MASS = None
        self.CENTRALS = None
        self.SATELLITES = None
        self.WEIGHTS = None
        return

    def resetHOD(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galaxies = np.zeros(len(self.massBins)-1,dtype=float)
        self.galaxies2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.centrals = np.zeros(len(self.massBins)-1,dtype=float)
        self.centrals2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.satellites = np.zeros(len(self.massBins)-1,dtype=float)
        self.satellites2 = np.zeros(len(self.massBins)-1,dtype=float)
        self.halos = np.zeros(len(self.massBins)-1,dtype=float)
        return

    def addHalosToHOD_deprecated(self,galaxies,massName="nodeMass200.0",weightHalos=False,mask=None,verbose=False):
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
        # Check if weights provided
        if "weight" in galaxies.dtype.names and weightHalos:
            weights = galaxies.weight
        else:
            weights = np.ones_like(galaxies[massName])
        weights = np.ones_like(galaxies[massName])
        # Find host halos
        CENTRALS = np.copy(galaxies.nodeIsIsolated==1).astype(bool)
        SATELLITES = np.copy(galaxies.nodeIsIsolated==0).astype(bool)
        totalHostHalos = len(galaxies.nodeIndex[CENTRALS])
        if verbose:
            print(funcname+"(): located "+str(totalHostHalos)+" host halos...")
        HOST = getHostIndex(galaxies.nodeIsIsolated)
        haloMass = np.log10(np.copy(galaxies[massName][HOST]))
        halos,bins = np.histogram(haloMass[CENTRALS],self.massBins,weights=weights[CENTRALS])
        self.halos += np.copy(halos.astype(float))
        # Count all galaxies
        if verbose:
            print(funcname+"(): counting galaxies...")
        galaxies,bins = np.histogram(haloMass[mask],self.massBins,weights=weights[mask])
        self.galaxies += np.copy(galaxies.astype(float))
        #self.galaxies2 += np.copy(galaxies**2).astype(float)
        self.galaxies2 += np.copy(np.sqrt(galaxies)).astype(float)
        # Count central galaxies
        if verbose:
            print(funcname+"(): counting central galaxies...")
        galaxies,bins = np.histogram(haloMass[np.logical_and(mask,CENTRALS)],self.massBins,\
                                     weights=weights[np.logical_and(mask,CENTRALS)])
        self.centrals += np.copy(galaxies.astype(float))
        #self.centrals2 += np.copy(galaxies**2).astype(float)
        self.centrals2 += np.copy(np.sqrt(galaxies)).astype(float)
        # Count satellite galaxies
        if verbose:
            print(funcname+"(): counting satellite galaxies...")
        galaxies,bins = np.histogram(haloMass[np.logical_and(mask,SATELLITES)],self.massBins,\
                                     weights=weights[np.logical_and(mask,SATELLITES)])
        self.satellites += np.copy(galaxies.astype(float))
        #self.satellites2 += np.copy(galaxies**2).astype(float)
        self.satellites2 += np.copy(np.sqrt(galaxies)).astype(float)
        return

    def addHalos(self,galaxies,massName="nodeMass200.0",weightHalos=False,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        galaxies = galaxies.view(np.recarray)
        # Check require properties are present
        required = [massName,"nodeIsIsolated","nodeIndex","parentIndex"]
        if not set(required).issubset(galaxies.dtype.names):
            missing = list(set(required).difference(galaxies.dtype.names))
            raise KeyError(funcname+"(): Following required properties not found: "+",".join(missing)+".")
        self.NAME = massName
        # Check if weights provided
        if "weight" in galaxies.dtype.names and weightHalos:
            self.WEIGHTS = galaxies.weight
        else:
            self.WEIGHTS = np.ones_like(galaxies[massName])
        # Find host halos
        self.CENTRALS = np.copy(galaxies.nodeIsIsolated==1).astype(bool)
        self.SATELLITES = np.copy(galaxies.nodeIsIsolated==0).astype(bool)
        totalHostHalos = len(galaxies.nodeIndex[self.CENTRALS])
        if verbose:
            print(funcname+"(): located "+str(totalHostHalos)+" host halos...")
        hosts = getHostIndex(galaxies.nodeIsIsolated)
        self.MASS = np.log10(np.copy(galaxies[massName][hosts]+1.0e-50))
        return

    def addHalosToHOD(self,mask=None,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check mask is provided -- if not then provide simple mask
        if mask is None:
            mask = np.ones_like(self.MASS)
        # Count number of host halos
        halos,bins = np.histogram(self.MASS[self.CENTRALS],self.massBins,weights=self.WEIGHTS[self.CENTRALS])
        self.halos += np.copy(halos.astype(float))
        # Exit here if no galaxies satisfy the mask
        if not any(mask):
            return
        # Count all galaxies
        if verbose:
            print(funcname+"(): counting galaxies...")
        galaxies,bins = np.histogram(self.MASS[mask],self.massBins,weights=self.WEIGHTS[mask])
        self.galaxies += np.copy(galaxies.astype(float))
        self.galaxies2 += np.copy(galaxies**2).astype(float)
        # Count central galaxies
        if verbose:
            print(funcname+"(): counting central galaxies...")
        galaxies,bins = np.histogram(self.MASS[np.logical_and(mask,self.CENTRALS)],self.massBins,\
                                         weights=self.WEIGHTS[np.logical_and(mask,self.CENTRALS)])
        self.centrals += np.copy(galaxies.astype(float))
        self.centrals2 += np.copy(galaxies**2).astype(float)
        # Count satellite galaxies
        if verbose:
            print(funcname+"(): counting satellite galaxies...")
        galaxies,bins = np.histogram(self.MASS[np.logical_and(mask,self.SATELLITES)],self.massBins,\
                                         weights=self.WEIGHTS[np.logical_and(mask,self.SATELLITES)])
        self.satellites += np.copy(galaxies.astype(float))
        self.satellites2 += np.copy(galaxies**2).astype(float)
        return


    def computeHOD(self,errors="stddev"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        np.place(self.halos,self.halos==0.0,1.0)
        dm = self.massBins[1] - self.massBins[0]
        bins = self.massBins[:-1] + dm/2.0
        dtype = [("mass",float),("allGalaxies",float),("allGalaxiesError",float),\
                 ("centrals",float),("centralsError",float),\
                 ("satellites",float),("satellitesError",float)]
        hod = np.zeros(len(bins),dtype=dtype).view(np.recarray)
        hod.mass = np.copy(bins)
        mean = self.galaxies/self.halos
        hod.allGalaxies = np.copy(mean)
        if fnmatch.fnmatch(errors.lower(),"stddev"):
            sigma = np.sqrt((self.galaxies2/self.halos)-mean**2)
        elif fnmatch.fnmatch(errors.lower(),"poisson"):
            sigma = np.sqrt(self.galaxies)/self.halos
        else:
            raise ValueError("Errors option not recognised! Options = 'stddev' or 'poisson'.")
        hod.allGalaxiesError = np.copy(sigma)
        mean = self.centrals/self.halos
        hod.centrals = np.copy(mean)
        if fnmatch.fnmatch(errors.lower(),"stddev"):
            sigma = np.sqrt((self.centrals2/self.halos)-mean**2)
        elif fnmatch.fnmatch(errors.lower(),"poisson"):
            sigma = np.sqrt(self.centrals)/self.halos
        else:
            raise ValueError("Errors option not recognised! Options = 'stddev' or 'poisson'.")
        hod.centralsError = np.copy(sigma)
        mean = self.satellites/self.halos
        hod.satellites = np.copy(mean)
        if fnmatch.fnmatch(errors.lower(),"stddev"):
            sigma = np.sqrt((self.satellites2/self.halos)-mean**2)
        elif fnmatch.fnmatch(errors.lower(),"poisson"):
            sigma = np.sqrt(self.satellites)/self.halos
        else:
            raise ValueError("Errors option not recognised! Options = 'stddev' or 'poisson'.")
        hod.satellitesError = np.copy(sigma)        
        return hod

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
