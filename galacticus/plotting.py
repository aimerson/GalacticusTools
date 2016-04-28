#! /usr/bin/env python

import sys
import numpy as np
import matplotlib
from scipy.stats import *

def X11_forwarding():
    import os
    try:
        l = os.environ["DISPLAY"]
    except KeyError:
        return False
    else:
        return True

if not X11_forwarding():
    matplotlib.use('Agg')
    print "WARNING! No X-window detected! Adopting 'Agg' backend..."
from pylab import *
import matplotlib.pyplot as plt
from decimal import *

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.numpoints'] = 1
#mpl.rcParams['legend.scatterpoints'] = 1
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.markerscale'] = 1
mpl.rcParams['axes.labelsize'] = 12.0
mpl.rcParams['xtick.labelsize'] = 12.0
mpl.rcParams['ytick.labelsize'] = 12.0

import sys,fnmatch
import numpy as np
from .io import GalacticusHDF5

###########################################################################
# MISC. FUNCTIONS
###########################################################################

def Legend(ax,ec='none',fc='none',fontcolor="k",**kwargs):
    leg = ax.legend(**kwargs)
    frame = leg.get_frame()
    frame.set_edgecolor(ec)
    frame.set_facecolor(fc)
    for text in leg.get_texts():
        text.set_color(fontcolor)
    return

def minor_ticks(axObj):
    """ 
    minor_ticks(): display minor tick marks on plot.

    USAGE: minor_ticks(axObj)

           axObj : axis object e.g. ax or ax.xaxis or ax.yaxis (If ax, will
                  plot ticks on both X and Y axes)
    """
    if str(axObj).startswith("Axes"):
        axObj =[axObj.xaxis,axObj.yaxis]
    else:
        axObj =[axObj]
    for ax in axObj:
        # Determine major tick intervals                                                                                                                                                                                    
        major_tick_locations = ax.get_majorticklocs()
        major_tick_interval = major_tick_locations[1] - major_tick_locations[0]
        # Create dummy figure                                                                                                                                                                                               
        dummyfig = plt.figure()
        dummyax = dummyfig.add_subplot(111)
        # Create dummy data over range of major tick interval                                                                                                                                                               
        dummyx = np.arange(0,np.fabs(major_tick_interval) + \
                               np.fabs(major_tick_interval)/10.0, \
                               np.fabs(major_tick_interval)/10.0)
        dummyax.plot(dummyx,dummyx)
        # Get minor tick interval by using automatically generated                                                                                                                                                          
        # major tick intervals from dummy plot                                                                                                                                                                              
        minor_tick_locations = dummyax.xaxis.get_majorticklocs()
        minor_tick_interval = minor_tick_locations[1] - minor_tick_locations[0]
        plt.close(dummyfig)
        ax.set_minor_locator(MultipleLocator(base=minor_tick_interval))
    return


###########################################################################
# GLOBAL HISTORY PLOTS
###########################################################################

def plot_global_history(ifile,ofile=None,SIunits=False,xunit="redshift"):
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = "/".join(ifile.split("/")[:-1]) + "globalHistory.pdf"

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
    print(funcname+"(): Plot output to file: "+ofile)




















