#! /usr/bin/env python


import numpy as np
from galacticus.plotting.utils import *

from galacticus.statistics.LuminosityFunctions.analyticFits import EuclidLuminosityFunction,SchechterLuminosities
from galacticus.statistics.LuminosityFunctions.observations import Halpha
from mpl_toolkits.axes_grid1 import AxesGrid


G95 = Halpha('Gallego95')
S09 = Halpha('Shim09',hubble=0.71)
C13 = Halpha('Colbert13')
S13 = Halpha('Sobral13')
GAMA = Halpha('Gunawardhana13_GAMA')
SDSS = Halpha('Gunawardhana13_SDSS')
    
luminosities = np.linspace(38,44,100)

fig = figure(figsize=(15,10))
axes = AxesGrid(fig,111,nrows_ncols=(2,3),axes_pad=0.0,\
                share_all=True,label_mode="L",\
                cbar_mode=None,aspect=False)
redshifts  = [0.09,0.4,0.84,1.1,1.48,2.23]

addLegend = True
hPozzetti = 0.7

for i,z in enumerate(redshifts):
    
    ax = axes[i]

    ls = ["dotted","dashed","dashdot"]
    for j in range(3):        
        EM = EuclidLuminosityFunction(j+1)
        phi = np.log10(EM.phi(10**luminosities,z))
        if z > 0.6 and addLegend:
            label = "Euclid Model "+str(j+1)
            if j == 2:
                addLegend = False
        else:
            label = None
        if j < 2:
            ax.plot(luminosities,phi,label=label,c='k',ls=ls[j],lw=2.0)
        else:
            if z > 0.6:
                ax.plot(luminosities,phi,label=label,c='k',ls=ls[j],lw=2.0)
    
    data = G95.selectRedshift(z)
    if data is not None:
        S = G95.Schechter(z)
        phi = np.log10(S.phi(10.**luminosities))
        ax.plot(luminosities,phi,c='g')
        l = data.log10L 
        phi = data.phi
        err = data.phiErr
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)
        np.place(negErr,np.isnan(negErr),99.0)
        ax.errorbar(l,np.log10(phi),yerr=[negErr,posErr],marker='^',c='g',ls='none',mfc='none',mec='g',label=G95.dataset)

    data = S09.selectRedshift(z)
    if data is not None:
        l = data.log10L 
        phi = data.phi
        err = data.phiErr
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)
        np.place(negErr,np.isinf(negErr),99.0)
        np.place(negErr,np.isnan(negErr),99.0)
        ax.errorbar(l,np.log10(phi),yerr=[negErr,posErr],marker='*',c='orange',ls='none',mfc='none',mec='orange',label=S09.dataset)

    data = C13.selectRedshift(z)
    if data is not None:
        S = C13.Schechter(z)
        phi = np.log10(S.phi(10.**luminosities))
        ax.plot(luminosities,phi,c='r')
        l = data.log10L
        phi = data.phiCorr
        err = data.phiCorrErr
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)
        np.place(negErr,np.isnan(negErr),99.0)
        ax.errorbar(l,np.log10(phi),yerr=[negErr,posErr],marker='s',c='r',ls='none',mfc='none',mec='r',label=C13.dataset)
        
    data = S13.selectRedshift(z)    
    if data is not None:
        S = S13.Schechter(z)
        phi = np.log10(S.phi(10.**luminosities))
        ax.plot(luminosities,phi,c='b')
        l = data.log10L+ np.log10((S13.hubble/0.7)**2)
        phi = data.phiCorr*((0.7/S13.hubble)**3)
        err = data.phiCorrErr*((0.7/S13.hubble)**3)
        posErr = np.log10(phi+err) - np.log10(phi)
        negErr = np.log10(phi) - np.log10(phi-err)
        np.place(negErr,np.isnan(negErr),99.0)
        xerr = data.errlog10L
        ax.errorbar(l,phi,yerr=err,xerr=xerr,marker='o',c='b',ls='none',mfc='none',mec='b',label=S13.dataset)

    data = GAMA.selectRedshift(z)    
    if data is not None:
        l = data.log10L
        phi = data.logphi
        posErr = data.logphiPosErr
        negErr = data.logphiNegErr
        np.place(negErr,np.isnan(negErr),99.0)
        ax.errorbar(l,phi,yerr=[negErr,posErr],marker='d',c='r',ls='none',mfc='none',mec='r',label=GAMA.dataset+" GAMA")

    data = SDSS.selectRedshift(z)    
    if data is not None:
        l = data.log10L
        phi = data.logphi
        posErr = data.logphiPosErr
        negErr = data.logphiNegErr
        np.place(negErr,np.isnan(negErr),99.0)
        ax.errorbar(l,phi,yerr=[negErr,posErr],marker='d',c='r',ls='none',mfc='r',mec='r',label=GAMA.dataset+" SDSS")

    ax.set_xlim(38.8,43.8)
    ax.set_ylim(-6,-1.01)
    minor_ticks(ax)

    ax.set_xlabel("$\log_{10}(L_{\mathrm{H\\alpha}}/\mathrm{erg}\,\,\mathrm{s}^{-1})$")
    ax.set_ylabel("$\log_{10}(\phi(L)/\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$")

    xpos,ypos = get_position(ax,0.9,0.9)
    ax.text(xpos,ypos,"z = "+str(z),ha='right',va='center')

    Legend(ax,loc=3)
    

savefig("Halpha_LF.pdf",bbox_inches='tight')




