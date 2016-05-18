#! /usr/bin/env python

import fnmatch
import numpy as np
from .io import GalacticusHDF5


def availableLines(f,z,frame=None,componentGalaxy=None,dust=None):
    if type(f) is GalacticusHDF5:
        allprops = G.availableDatasets(z)
    else:
        G = GalacticusHDF5(f,"r")
        allprops = G.availableDatasets(z)
        G.close()
    # Select component to search for
    if componentGalaxy is None:
        search = "*LineLuminosity:"
    else:
        if componentGalaxy.lower() == "total":
            search = "totalLineLuminosity:"
        elif componentGalaxy.lower() == "disk":
            search = "diskLineLuminosity:"
        elif componentGalaxy.lower() == "spheroid":
            search = "spheroidLineLuminosity:"    
    search = search + "*"    
    # Select frame to search for
    if frame is None:
        search = search + ":*:"
    else:
        if frame.lower() == "rest":
            search = search + ":rest:"
        elif frame.lower() == "observed":
            search = search + ":observed:"
    search = search + "*z*[0-9]"
    # Select whether include dust
    if dust is None:
        search = search + "*"
    else:
        if dust:
            search = search + ":dustAtlas"
    # Search properties
    avail = fnmatch.filter(allprops,search)
    avail = [ l.split(":")[1] for l in avail ]
    return list(np.unique(avail))


def getLatexName(line):    
    name = "\mathrm{line}"
    if line.lower() == "balmeralpha6563":
        name = "\mathrm{H\\alpha}"
    elif line.lower() == "balmerbeta4861":
        name = "\mathrm{H\beta}"
    else:
        ones = "".join(fnmatch.filter(list(line),"I"))
        wave = line.split(ones)[-1]
        elem = line.split(ones)[0][0].upper()
        name = "\mathrm{"+elem+ones+"_{"+wave+"\\AA}}"
    return name




