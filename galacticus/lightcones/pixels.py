#! /usr/bin/env python

import numpy as np
import healpy as hp

from ..io import GalacticusHDF5
from ..constants import Pi
from ..utils.progress import Progress


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
        return
    
    
    def selectGalaxiesInPixel(self,ra,dec,pixelNumber):        
        pixels = hp.ang2pix(self.NSIDE,dec,ra,nest=self.nest,lonlat=True)
        return pixels==pixelNumber
    
    
    
    


    
