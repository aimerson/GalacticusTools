#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from utils_plotting import *
from galacticus import *


ifile = None
ofile = None
xunit = "redshift"
SIunits = False
iarg = 0
while iarg < len(sys.argv):
    if fnmatch.fnmatch(sys.argv[iarg],"-i*"):
        iarg += 1
        ifile = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
        iarg += 1
        ofile = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg].lower(),"-si"):
        SIunits = True
    if fnmatch.fnmatch(sys.argv[iarg],"-x"):
        iarg += 1
        xunit = sys.argv[iarg].lower()
    iarg += 1


G = GalacticusHDF5(ifile,'r')
history = G.global_history(si=SIunits)

if xunit in ["z","redshift"]:
    x = history.historyRedshift
    xlabel = "$z$"
elif xunit in ["a","expansion"]:
    x = history.historyExpansion
    xlabel = "$a$"
elif xunit in ["t","time"]:
    x = history.historyTime    
    if SIunits:
        factor = int(np.min(np.floor(np.log10(x))))
        x /= 10.0**factor
        xlabel = "$t\,[10^{"+str(factor)+"}\,\mathrm{s}]$"
    else:
        xlabel = "$t\,[\mathrm{Gyr}]$"


fig = figure(figsize=(18,6))

# Matter density
ax = fig.add_subplot(131,yscale='log')
ax.plot(x,history.historyNodeDensity,ls=':',c='k',lw=2.5,label="Halos")
ax.plot(x,history.historyHotGasDensity,ls='-',c='r',label="Hot Gas")
ax.plot(x,history.historyGasDensity,ls='--',c='b',label="Cooled Gas")
ax.plot(x,history.historyStellarDensity,ls='-.',c='g',lw=2.5,label="Stars")
Legend(ax,loc=0,title="MATTER DENSITY")
minor_ticks(ax.xaxis)
ax.set_xlabel(xlabel,fontsize=14)
if SIunits:
    ax.set_ylabel("$\log_{10}(\\rho/\mathrm{kg}\,\,\mathrm{m}^{-3})$")
else:
    ax.set_ylabel("$\log_{10}(\\rho/\mathrm{M_{\odot}}\,\,\mathrm{Mpc}^{-3})$")

# Stellar density
ax = fig.add_subplot(132,yscale='log')
ax.plot(x,history.historyStellarDensity,ls=':',c='k',lw=2.5,label="Total")
ax.plot(x,history.historyDiskStellarDensity,ls='-',c='b',label="Disks")
ax.plot(x,history.historySpheroidStellarDensity,ls='--',c='r',label="Spheroids")
Legend(ax,loc=0,title="STELLAR DENSITY")
minor_ticks(ax.xaxis)
ax.set_xlabel(xlabel,fontsize=14)
if SIunits:
    ax.set_ylabel("$\log_{10}(\\rho_{\star}/\mathrm{kg}\,\,\mathrm{m}^{-3})$")
else:
    ax.set_ylabel("$\log_{10}(\\rho_{\star}/\mathrm{M_{\odot}}\,\,\mathrm{Mpc}^{-3})$")


# Star formation rate
ax = fig.add_subplot(133,yscale='log')
ax.plot(x,history.historyStarFormationRate,ls=':',c='k',lw=2.5,label="Total")
ax.plot(x,history.historyDiskStarFormationRate,ls='-',c='b',label="Disks")
ax.plot(x,history.historySpheroidStarFormationRate,ls='--',c='r',label="Spheroids")
Legend(ax,loc=0,title="STAR FORMATION RATE")
minor_ticks(ax.xaxis)
ax.set_xlabel(xlabel,fontsize=14)
if SIunits:
    ax.set_ylabel("$\log_{10}(\dot{\\rho}_{\star}/\mathrm{kg}\,\,\mathrm{s}^{-1}\,\,\mathrm{m}^{-3})$")
else:
    ax.set_ylabel("$\log_{10}(\dot{\\rho}_{\star}/\mathrm{M_{\odot}}\,\,\mathrm{Gyr}^{-1}\,\,\mathrm{Mpc}^{-3})$")

savefig(ofile,bbox_inches='tight')





















