#! /usr/bin/env python


import sys,fnmatch
import numpy as np
from .utils import *
from ..io import GalacticusHDF5
from ..constants import luminositySolar,erg,massSolar

###########################################################################
# GLOBAL HISTORY PLOTS
###########################################################################

def plotGlobalHistory(ifile,ofile=None,SIunits=False,xunit="redshift"):
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = ifile.split("/")
        ofile[-1] = "globalHistory.pdf"
        ofile = "/".join(ofile)
    G = GalacticusHDF5(ifile,'r')
    history = G.global_history(si=SIunits)
    G.close()

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
        ax.set_ylabel("$\\rho\,\,[\mathrm{kg}\,\,\mathrm{m}^{-3}]$")
    else:
        ax.set_ylabel("$\\rho\,\,[\mathrm{M_{\odot}}\,\,\mathrm{Mpc}^{-3}]$")

    # Stellar density
    ax = fig.add_subplot(132,yscale='log')
    ax.plot(x,history.historyStellarDensity,ls=':',c='k',lw=2.5,label="Total")
    ax.plot(x,history.historyDiskStellarDensity,ls='-',c='b',label="Disks")
    ax.plot(x,history.historySpheroidStellarDensity,ls='--',c='r',label="Spheroids")
    Legend(ax,loc=0,title="STELLAR DENSITY")
    minor_ticks(ax.xaxis)
    ax.set_xlabel(xlabel,fontsize=14)
    if SIunits:
        ax.set_ylabel("$\\rho_{\star}\,\,[\mathrm{kg}\,\,\mathrm{m}^{-3}]$")
    else:
        ax.set_ylabel("$\\rho_{\star}\,\,[\mathrm{M_{\odot}}\,\,\mathrm{Mpc}^{-3}]$")
                
    # Star formation rate
    ax = fig.add_subplot(133,yscale='log')
    ax.plot(x,history.historyStarFormationRate,ls=':',c='k',lw=2.5,label="Total")
    ax.plot(x,history.historyDiskStarFormationRate,ls='-',c='b',label="Disks")
    ax.plot(x,history.historySpheroidStarFormationRate,ls='--',c='r',label="Spheroids")
    Legend(ax,loc=0,title="STAR FORMATION RATE")
    minor_ticks(ax.xaxis)
    ax.set_xlabel(xlabel,fontsize=14)
    if SIunits:
        ax.set_ylabel("$\dot{\\rho}_{\star}\,\,[\mathrm{kg}\,\,\mathrm{s}^{-1}\,\,\mathrm{m}^{-3}]$")
    else:
        ax.set_ylabel("$\dot{\\rho}_{\star}\,\,[\mathrm{M_{\odot}}\,\,\mathrm{Gyr}^{-1}\,\,\mathrm{Mpc}^{-3}]$")

    # Save figure and return
    savefig(ofile,bbox_inches='tight')
    print(funcname+"(): Plot output to file: "+ofile)
    return

