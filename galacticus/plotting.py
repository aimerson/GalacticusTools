#! /usr/bin/env python

import sys,fnmatch
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


def print_rounded_value(x,dx):
    return str(Decimal(str(x)).quantize(Decimal(str(dx))))

def sigfig(x,n,latex=True):
    if n ==0:
        return 0
    fmt = "%."+str(n-1)+"E"
    s = fmt % x
    s = str(float(s))
    if "e" in s:
        s = s.split("e")
        m = n - len(s[0].replace(".","").replace("-","").lstrip("0"))
        s[0] = s[0].ljust(len(s[0])+m,"0")
        if "." in s[0]:
            if len(s[0].split(".")[1].strip("0")) == 0 and len(s[0].split(".")[0]) >= n:
                s[0] = s[0].split(".")[0]
        if latex:
                s[1] = "$\\times 10^{"+str(int(s[1]))+"}$"
        else:
            s[1] = "e" + s[1]
        s = "".join(s)
    else:
        m = n - len(s.replace(".","").replace("-","").lstrip("0"))
        s = s.ljust(len(s)+m,"0")
        if "." in s:
            if len(s.split(".")[1].strip("0")) == 0 and len(s.split(".")[0]) >= n:
                s = s.split(".")[0]
    return s


def get_position(ax,xfrac,yfrac):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    dx = float(xlims[1]) - float(xlims[0])
    dy = float(ylims[1]) - float(ylims[0])
    xpos = float(xlims[0]) + xfrac*dx
    ypos = float(ylims[0]) + yfrac*dy
    return xpos,ypos

###########################################################################
# GLOBAL HISTORY PLOTS
###########################################################################

def plot_global_history(ifile,ofile=None,SIunits=False,xunit="redshift"):
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = "/".join(ifile.split("/")[:-1]) + "/globalHistory.pdf"

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


###########################################################################
# MASS & LUMINOSITY FUNCTIONS
###########################################################################


def plot_stellar_mass_function(ifile,z,ofile=None,mbins=None,disks=False,spheroids=False):
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = "/".join(ifile.split("/")[:-1]) + "/stellarMassFunction_z"+str(z).replace(".","p")+".pdf"    
    
    # Read galaxies information
    G = GalacticusHDF5(ifile,'r')
    props = ["massStellar","weight"]
    if disks:
        props.append("diskMassStellar")
    if spheroids:
        props.append("spheroidMassStellar")        
    galaxies = G.galaxies(props=props,z=z)
    G.close()

    # Set mass bins
    if mbins is None:
        mbins = np.arange(8.0,12.0,0.1)
    dM = mbins[1] - mbins[0]
    
    # Create figure
    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111,yscale='log')
    
    # Create axis labels
    fs = 14
    ax.set_xlabel("$\log_{10}(M_{\star}\,/\,\mathrm{M_{\odot}})$",fontsize=fs)
    ax.set_ylabel("$\mathrm{d}N(M_{\star})/\mathrm{d}\log_{10}M_{\star}\,\,[\mathrm{Mpc}^{-3}]$",fontsize=fs)

    # Plot mass function for total stellar mass
    mass = np.log10(galaxies["massStellar"])
    wgts = galaxies["weight"]
    mf,bins = np.histogram(mass,bins=mbins,weights=wgts)
    bins = bins[:-1] + dM/2.0
    ax.plot(bins,mf,c='k',ls='-',label="Galacticus (Total)")

    # Plot mass function for stellar mass in disks
    if disks:
        mass = np.log10(galaxies["diskMassStellar"])
        wgts = galaxies["weight"]
        mf,bins = np.histogram(mass,bins=mbins,weights=wgts)
        bins = bins[:-1] + dM/2.0
        ax.plot(bins,mf,c='b',ls='--',label="Galacticus (Disks)")

    # Plot mass function for stellar mass in spheroids
    if spheroids:
        mass = np.log10(galaxies["spheroidMassStellar"])
        wgts = galaxies["weight"]
        mf,bins = np.histogram(mass,bins=mbins,weights=wgts)
        bins = bins[:-1] + dM/2.0
        ax.plot(bins,mf,c='r',ls=':',lw=2.5,label="Galacticus (Spheroids)")
    
    minor_ticks(ax.xaxis)
    Legend(ax,loc=0,title="STELLAR MASS\nFUNCTION ($z\,=\,"+sigfig(z,2)+"$)")
    
    # Save figure and return
    savefig(ofile,bbox_inches='tight')
    print(funcname+"(): Plot output to file: "+ofile)    
    return



def plot_halpha_luminosity_function(ifile,z,ofile=None,lbins=None,disks=False,spheroids=False,dust=False,frame="rest",ergs=False):    
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = "/".join(ifile.split("/")[:-1]) + "/HalphaLuminosityFunction_z"+str(z).replace(".","p")+".pdf"    
    
    # Solar luminosity in erg/s
    Lsol = 3.828e33 
    factor = 40

    # Set properties to read
    search = "totalLineLuminosity:balmerAlpha6563:"
    if fnmatch.fnmatch(frame.lower(),"r*"):
        search = search + "rest*"
    elif fnmatch.fnmatch(frame.lower(),"o*"):
        search = search + "observed*"
    if disks or spheroids:
        search = search.replace("totalLine","*Line")

    # Read galaxies information
    G = GalacticusHDF5(ifile,'r')
    props = ["weight",search]
    galaxies = G.galaxies(props=props,z=z)
    G.close()    

    # Set luminosity bins
    if lbins is None:
        lbins = np.arange(-1.0,8.0,0.5)
        if ergs:
            lbins += np.log10(Lsol) - factor
    dL = lbins[1] - lbins[0]

    # Create figure
    fig = figure(figsize=(6,6))
    ax = fig.add_subplot(111,yscale='log')

    # Create axis labels
    fs = 14
    if ergs:
        xlabel = "$\log_{10}(L_{\mathrm{H\\alpha}}\,/\,10^{"+str(factor)+"}\,\mathrm{erg}\,\mathrm{s}^{-1})$"        
    else:
        xlabel = "$\log_{10}(L_{\mathrm{H\\alpha}}\,/\,\mathrm{L_{\odot}})$"
    ax.set_xlabel(xlabel,fontsize=fs)
    ylabel = "$\mathrm{d}\phi(L_{\mathrm{H\\alpha}})/\mathrm{d}\log_{10}L_{\mathrm{H\\alpha}}\,\,[\mathrm{Mpc}^{-3}]$"
    ax.set_ylabel(ylabel,fontsize=fs)

    # Plot luminosity function for total Halpha luminosity
    name = fnmatch.filter(galaxies.dtype.names,"totalLine*")[0]    
    lum = np.log10(galaxies[name])
    if ergs:
        lum += np.log10(Lsol) - factor
    wgts = galaxies["weight"]
    lf,bins = np.histogram(lum,bins=lbins,weights=wgts)
    bins = bins[:-1] + dL/2.0
    ax.plot(bins,lf,c='k',ls='-',label="Galacticus (Total)")
    
    # Plot luminosity function for disk Halpha luminosity
    if disks:
        name = fnmatch.filter(galaxies.dtype.names,"diskLine*")[0]    
        lum = np.log10(galaxies[name])
        if ergs:
            lum += np.log10(Lsol) - factor
        wgts = galaxies["weight"]
        lf,bins = np.histogram(lum,bins=lbins,weights=wgts)
        bins = bins[:-1] + dL/2.0
        ax.plot(bins,lf,c='b',ls='--',label="Galacticus (Disks)")
    

    # Plot luminosity function for spheroid Halpha luminosity
    if spheroids:
        name = fnmatch.filter(galaxies.dtype.names,"spheroidLine*")[0]    
        lum = np.log10(galaxies[name])
        if ergs:
            lum += np.log10(Lsol) - factor
        wgts = galaxies["weight"]
        lf,bins = np.histogram(lum,bins=lbins,weights=wgts)
        bins = bins[:-1] + dL/2.0
        ax.plot(bins,lf,c='r',ls=':',lw=2.5,label="Galacticus (Spheroids)")

    
    
    minor_ticks(ax.xaxis)
    Legend(ax,loc=0,title="$\mathrm{H\\alpha}$ LUMINOSITY\nFUNCTION ($z\,=\,"+sigfig(z,2)+"$)")
    


    # Save figure and return
    savefig(ofile,bbox_inches='tight')
    print(funcname+"(): Plot output to file: "+ofile)    
    return






