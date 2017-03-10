#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import healpy as hp

from ..io import GalacticusHDF5
from ..constants import Pi
from ..utils.progress import Progress
from .utils import getRaDec


def estimateNSIDE(files,totalArea,galaxiesPerPixel=10000):    
    # Loop over files to count total number of galaxies over all outputs
    def getGalaxyCount(fileName,progObj):
        GH5 = GalacticusHDF5(fileName,'r')
        galaxies = GH5.countGalaxies()
        GH5.close()
        progObj.increment()
        progObj.print_status_line(task="galaxies in file: "+str(galaxies))
        return galaxies
    PROG = Progress(len(files))
    totalGalaxies = np.sum(np.array([getGalaxyCount(fileName,PROG) for fileName in files]))
    # Compute mean number of galaxies per square degree
    galaxiesPerSquareDegree = float(totalGalaxies)/totalArea
    # Construct list of available NSIDE values
    nsides = 2**np.arange(14)
    areas = hp.nside2pixarea(nsides,degrees=True)
    # Find NSIDE with galaxies per pixel closest to desired value
    galaxies = np.array(areas*galaxiesPerSquareDegree).astype(int)
    iside = np.argmin(np.fabs(galaxies-galaxiesPerPixel))
    return nsides[iside]


class Pixels(object):
    
    def __init__(self,NSIDE,nest=False):
        self.NSIDE = NSIDE
        self.nest = nest
        self.galaxyCounts = np.zeros(hp.nside2npix(self.NSIDE))
        return

    def getGalaxyPixelNumbers(self,ra,dec):
        pixels = hp.ang2pix(self.NSIDE,dec,ra,nest=self.nest,lonlat=True)
        return pixels

    def selectGalaxiesInPixel(self,ra,dec,pixelNumber):        
        pixels = self.getGalaxyPixelNumbers(ra,dec)
        return pixels==pixelNumber




    def _addOutputToCounts(self,galHDF5Obj,z):
        datasets = galHDF5Obj.availableDatasets(z)
        if len(datasets) == 0:
            return
        datasets = fnmatch.filter(datasets,"lightconePosition[XYZ]")
        galaxies = galHDF5Obj.readGalaxies(z,props=datasets)
        galaxies = galaxies.view(np.recarray)
        ra,dec = getRaDec(galaxies.lightconePositionX,galaxies.lightconePositionY,galaxies.lightconePositionZ)
        pixels = self.getGalaxyPixelNumbers(ra,dec)
        pixelCounts = np.bincount(pixels,minlength=hp.nside2npix(self.NSIDE))
        self.galaxyCounts += pixelCounts
        return

    def _addHDF5FileToCounts(self,fileName,progObj=None):
        GH5 = GalacticusHDF5(fileName,'r')
        if GH5.outputs is not None:
            dummy = [self._addOutputToCounts(GH5,z) for z in GH5.outputs.z]
            del dummy
        GH5.close()
        if progObj is not None:
            progObj.increment()
            progObj.print_status_line()
        return

    
    def updateGalaxyCounts(self,files,fitsFile=None):
        PROG = Progress(len(files))
        dummy = [self._addHDF5FileToCounts(fileName,progObj=PROG) for fileName in files]
        if fitsFile is not None:
            if os.path.exists(fitsFile):
                os.remove(fitsFile)
            hp.write_map(fitsFile,self.galaxyCounts,nest=self.nest,coord='C')
        return


    
    
    
    


    
