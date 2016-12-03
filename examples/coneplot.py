#! /usr/bin/env python

from galacticus.plotting.utils import *
from galacticus.plotting.lightcones import conePlot

fig = figure(figsize=(10,8))

CONE = conePlot(fig,(0.0,6.0,0.5),(400.0,600.0,100.0),\
                    angularUnits="hour",angularSigFig=2,angularLabel="RA",\
                    radialUnits="Mpc",radialSigFig=2,radialLabel="$r\,\,$",\
                    subplot=211,rotate=90.0,gridlines=True,verbose=False)

a = np.random.rand(100)*90.0
r = np.random.rand(100)*200 + 400.0
CONE.aux_ax.scatter(a,r,facecolor='b',edgecolor='none')

CONE = conePlot(fig,(12.0,18.0,0.5),(400.0,600.0,100.0),\
                    angularUnits="hour",angularSigFig=2,angularLabel="RA",\
                    radialUnits="Mpc",radialSigFig=2,radialLabel="$r\,\,$",\
                    subplot=212,rotate=90.0,gridlines=False,verbose=False)

a = np.random.rand(500)*90.0 + 180.0
r = np.random.rand(500)*200 + 400.0
CONE.aux_ax.scatter(a,r,facecolor='none',edgecolor='r')

savefig("coneplot_example.pdf",bbox_inches='tight')

