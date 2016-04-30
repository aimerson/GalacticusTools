#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .utils import *
from ..io import GalacticusHDF5
from ..constants import luminositySolar,erg
from ..emlines import getLatexName

class LuminosityFunction(object):

    def __init__(self,figsize=(6,6),subplot=111,xlabel=None,ylabel=None,fontsize=12,**kwargs):
        self.fig = figure(figsize=figsize)
        self.ax = self.fig.add_subplot(subplot,**kwargs)
        self.fontsize = fontsize
        if xlabel is not None:
            self.ax.set_xlabel(xlabel,fontsize=self.fontsize)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel,fontsize=self.fontsize)
        return

    def addModel(self,mag,magbins,weights=None,**kwargs):
        dM = magbins[1] - magbins[0]
        lf,bins = np.histogram(mag,bins=magbins,weights=weights)
        bins = bins[:-1] + dM/2.0
        self.ax.plot(bins,lf,**kwargs)
        return (bins,lf)

    def output(self,ofile=None,show_legend=True,**kwargs):
        classname = self.__class__.__name__
        minor_ticks(self.ax.xaxis)
        if show_legend:
            Legend(self.ax,**kwargs)
        if ofile is None:
            # Display plot to screen
            show()
        else:
            # Save figure and return
            savefig(ofile,bbox_inches='tight')
            print(classname+"(): Plot output to file: "+ofile)
        return


def EmissionLine(ifile,z,line,ofile=None,lumbins=None,dust=False,frame="rest",ergs=False,disks=False,spheroids=False):
    funcname = sys._getframe().f_code.co_name    
    if ofile is None:
        ofile = "/".join(ifile.split("/")[:-1]) + "/EmLineLuminosityFunction_z"+str(z).replace(".","p")+".pdf"    
        
    # Solar luminosity in erg/s
    Lsol = luminositySolar/erg
    factor = 40
    
    # Set properties to read
    search = "*LineLuminosity:"+line+":"
    if fnmatch.fnmatch(frame.lower(),"r*"):
        search = search + "rest*"
    elif fnmatch.fnmatch(frame.lower(),"o*"):
        search = search + "observed*"

    # Read galaxies information
    G = GalacticusHDF5(ifile,'r')
    props = ["weight",search]
    galaxies = G.galaxies(props=props,z=z)
    G.close()    

    # Set luminosity bins
    if lumbins is None:
        lumbins = np.arange(0.0,8.0,0.1)
        if ergs:
            lumbins += np.log10(Lsol) - factor

    # Create axis labels
    latex = getLatexName(line)
    fs = 14
    if ergs:
        xlabel = "$\log_{10}(L_{"+latex+"}\,/\,10^{"+str(factor)+"}\,\mathrm{erg}\,\mathrm{s}^{-1})$"        
    else:
        xlabel = "$\log_{10}(L_{"+latex+"}\,/\,\mathrm{L_{\odot}})$"
    ylabel = "$\mathrm{d}\phi(L_{"+latex+"})/\mathrm{d}\log_{10}L_{"+latex+"}\,\,[\mathrm{Mpc}^{-3}]$"
        
    # Inititalise LuminosityFunction class
    LF = LuminosityFunction(yscale='log',xlabel=xlabel,ylabel=ylabel,fontsize=fs)
    
    # Plot luminosity function for total line luminosity
    name = fnmatch.filter(galaxies.dtype.names,"totalLine*")[0]    
    lum = np.log10(galaxies[name])
    if ergs:
        lum += np.log10(Lsol) - factor
    wgts = galaxies["weight"]
    LF.addModel(lum,lumbins,weights=wgts,c='k',ls='-',label="Galacticus (Total)")
    
    # Plot luminosity function for disk line luminosity
    if disks:
        name = fnmatch.filter(galaxies.dtype.names,"diskLine*")[0]    
        lum = np.log10(galaxies[name])
        if ergs:
            lum += np.log10(Lsol) - factor
        wgts = galaxies["weight"]
        LF.addModel(lum,lumbins,weights=wgts,c='b',ls='--',label="Galacticus (Disks)")

    # Plot luminosity function for spheroid line luminosity
    if spheroids:
        name = fnmatch.filter(galaxies.dtype.names,"spheroidLine*")[0]    
        lum = np.log10(galaxies[name])
        if ergs:
            lum += np.log10(Lsol) - factor
        wgts = galaxies["weight"]
        LF.addModel(lum,lumbins,weights=wgts,c='r',ls=':',lw=2.5,label="Galacticus (Spheroids)")
    
    # Output plot and return
    LF.output(ofile,loc=0,title="$"+latex+"$ LUMINOSITY\nFUNCTION ($z\,=\,"+sigfig(z,2)+"$)")
    return    


