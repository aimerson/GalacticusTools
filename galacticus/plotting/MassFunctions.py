#! /usr/bin/env python

import sys,fnmatch
import numpy as np
from .utils import *
from ..io import GalacticusHDF5


class MassFunction(object):

    def __init__(self,figsize=(6,6),subplot=111,xlabel=None,ylabel=None,fontsize=12,**kwargs):
        self.fig = figure(figsize=figsize)
        self.ax = self.fig.add_subplot(subplot,**kwargs)
        self.fontsize = fontsize
        if xlabel is not None:
            self.ax.set_xlabel(xlabel,fontsize=self.fontsize)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel,fontsize=self.fontsize)
        return

    def addModel(self,mass,massbins,weights=None,**kwargs):
        dM = massbins[1] - massbins[0]
        mf,bins = np.histogram(mass,bins=massbins,weights=weights)
        bins = bins[:-1] + dM/2.0
        self.ax.plot(bins,mf,**kwargs)
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



def StellarMassFunction(ifile,z,ofile=None,massbins=None,disks=False,spheroids=False):
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
    if massbins is None:
        massbins = np.arange(8.0,12.0,0.1)

    # Initialise MassFunction class
    fs = 14
    xlabel = "$\log_{10}(M_{\star}\,/\,\mathrm{M_{\odot}})$"
    ylabel = "$\mathrm{d}N(M_{\star})/\mathrm{d}\log_{10}M_{\star}\,\,[\mathrm{Mpc}^{-3}]$"
    MF = MassFunction(xlabel=xlabel,ylabel=ylabel,fontsize=fs)

    # Plot mass function for total stellar mass
    mass = np.log10(galaxies["massStellar"])
    wgts = galaxies["weight"]
    MF.addModel(mass,massbins,weights=wgts,c='k',ls='-',label="Galacticus (Total)")

    # Plot mass function for stellar mass in disks
    if disks:
        mass = np.log10(galaxies["diskMassStellar"])
        wgts = galaxies["weight"]
        MF.addModel(mass,massbins,weights=wgts,c='b',ls='--',label="Galacticus (Disks)")

    # Plot mass function for stellar mass in spheroids
    if spheroids:
        mass = np.log10(galaxies["spheroidMassStellar"])
        wgts = galaxies["weight"]
        MF.addModel(mass,massbins,weights=wgts,c='r',ls=':',lw=2.5,label="Galacticus (Spheroids)")
    
    # Save figure and return
    MF.output(ofile,loc=0,title="STELLAR MASS\nFUNCTION ($z\,=\,"+sigfig(z,2)+"$)")
    return


