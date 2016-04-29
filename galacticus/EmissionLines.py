#! /usr/bin/env python

import fnmatch
import numpy as np
from .io import GalacticusHDF5




def availableLines(ifile,z,comp="Total"):
    G = GalacticusHDF5(ifile,"r")
    allprops = G.availableDatasets(z)
    print allprops
    return None


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




