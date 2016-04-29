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

####################################################################################


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

