#! /usr/bin/env python


import numpy as np
from galacticus.plotting.utils import *

from galacticus.statistics.LuminosityFunctions.analyticFits import SchechterMagnitudes
from galacticus.statistics.LuminosityFunctions.observations import PhotometricBand


fig = figure(figsize=(15,20))

subplots_adjust(wspace=0.02)

bands = "J H K u g r i z Y bJ NUV FUV".split()

for i,band in enumerate(bands):
    
    ax = fig.add_subplot(4,3,i+1)
    
    if band != "bJ":
        OBS = PhotometricBand("gama",band)
        mag = OBS.data.mag
        phi = OBS.data.phi
        err = OBS.data.phiErr
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)

        mask = np.invert(np.logical_or(np.isnan(negErr),np.isinf(negErr)))
        mag = mag[mask]
        phi = phi[mask]
        posErr = posErr[mask]
        negErr = negErr[mask]

        ax.errorbar(mag,np.log10(phi),yerr=[negErr,posErr],marker='o',c='k',ls='none',mfc='none',mec='k',label="GAMA")
    if band in "J K bJ".split():
        OBS = PhotometricBand("2dF",band)
        mag = OBS.data.mag
        phi = OBS.data.phi
        err = OBS.data.phiErr
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)
        
        mask = np.invert(np.logical_or(np.isnan(negErr),np.isinf(negErr)))
        mag = mag[mask]
        phi = phi[mask]
        posErr = posErr[mask]
        negErr = negErr[mask]
        ax.errorbar(mag,np.log10(phi),yerr=[negErr,posErr],marker='s',c='b',ls='none',mfc='none',mec='b',label="2dFGRS")
    if band == "K":
        OBS = PhotometricBand("2MASS",band)
        mag = OBS.data.mag
        phi = OBS.data.logphi
        err = OBS.data.logphiErr
        ax.errorbar(mag,phi,yerr=err,marker='^',c='g',ls='none',mfc='none',mec='g',label="2MASS")

        OBS = PhotometricBand("6dF",band)
        mag = OBS.data.mag
        phi = OBS.data.logphi
        posErr = OBS.data.logphiPosErr
        negErr = OBS.data.logphiNegErr        
        ax.errorbar(mag,phi,yerr=[negErr,posErr],marker='v',c='orange',ls='none',mfc='none',mec='orange',label="6dFGS")

    if i in range(3) or i in range(9,12):
        ax.set_ylim(bottom=-6.0,top=-0.5)
    else:
        ax.set_ylim(bottom=-5.5,top=-1)        
    ax.invert_xaxis()
    ax.set_xlim(left=-11)
    if band == "bJ":
        ax.set_xlim(-13,-23)
    if band == "J":
        ax.set_xlim(-13,-25)
    if band == "H":
        ax.set_xlim(-14.5,-24.5)
    if band == "K":
        ax.set_xlim(-15,-25)
    if band == "u":
        ax.set_xlim(-13,-21.95)
    if band == "g":
        ax.set_xlim(-13,-23.95)
    if band == "r":
        ax.set_xlim(-13,-23.95)
    if band == "i":
        ax.set_xlim(-13.75,-23.95)        
    if band == "z":
        ax.set_xlim(-14,-23.95)
    if band == "Y":
        ax.set_xlim(-14,-23.95)
    if band == "NUV":
        ax.set_xlim(-11,-20.5)
    if band == "FUV":
        ax.set_xlim(-11.05,-20)

    minor_ticks(ax)

    if band == "bJ":
        label = "b_J"
    else:
        label = band
    ax.set_xlabel("$M_{\mathrm{"+label+"}}\,-\,5\log_{10}(h)$")
    if i%3 == 0:
        ax.set_ylabel("$\log_{10}(\phi/h^{-3}\,\mathrm{Mpc}^{-3}\,\mathrm{mag}^{-1})$")
    else:
        setp(ax.yaxis.get_ticklabels(),visible=False)

    Legend(ax,loc=3)
    

savefig("z0_LF.pdf",bbox_inches='tight')




