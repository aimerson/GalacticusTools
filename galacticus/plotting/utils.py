#! /usr/bin/env python

import sys,fnmatch
import math,re
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

#mpl.style.use('classic')
mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['xtick.top'] = True
#mpl.rcParams['ytick.right'] = True
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.markerscale'] = 1
mpl.rcParams['axes.labelsize'] = 12.0
mpl.rcParams['xtick.labelsize'] = 12.0
mpl.rcParams['ytick.labelsize'] = 12.0

####################################################################################
# AXIS FUNCTIONS
####################################################################################

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


class axisGridArray(object):
    def __init__(self,nrow,ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.ntot = self.nrow*self.ncol
        self.axisNumbers = np.arange(self.ntot).reshape(self.nrow,self.ncol)
        self.yEdges = self.axisNumbers[:,0]
        self.xEdges = self.axisNumbers[-1,:]
        return
    
    def getRow(self,i):
        return self.axisNumbers[i,:]

    def getColumn(self,i):
        return self.axisNumbers[:,i]

    def yEdge(self,iplot):
        if type(iplot) == int or type(iplot) == str:
            iax = int(list(str(iplot))[-1])
        else:
            iax = int(iplot[-1])
        return iax-1 in self.yEdges

    def xEdge(self,iplot):
        if type(iplot) == int or type(iplot) == str:
            iax = int(list(str(iplot))[-1])
        else:
            iax = int(iplot[-1])
        return iax-1 in self.xEdges
        



def yEdgeAxes(iplot):
    if type(iplot) == int:
        iplot = str(iplot)
    if type(iplot) == str:
        nrow = int(list(iplot)[0])
        ncol = int(list(iplot)[1])    
    else:    
        nrow = int(iplot[0])
        ncol = int(iplot[1])
    ntot = nrow*ncol
    i = np.arange(ntot).reshape(nrow,ncol)
    return i[:,0]

def xEdgeAxes(iplot):
    if type(iplot) == int:
        iplot = str(iplot)
    if type(iplot) == str:
        nrow = int(list(iplot)[0])
        ncol = int(list(iplot)[1])    
    else:    
        nrow = int(iplot[0])
        ncol = int(iplot[1])
    ntot = nrow*ncol
    i = np.arange(ntot).reshape(nrow,ncol)
    return i[-1,:]


def get_position(ax,xfrac,yfrac):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    dx = float(xlims[1]) - float(xlims[0])
    dy = float(ylims[1]) - float(ylims[0])
    xpos = float(xlims[0]) + xfrac*dx
    ypos = float(ylims[0]) + yfrac*dy
    return xpos,ypos


####################################################################################
# COLOURS FUNCTIONS
####################################################################################

def colour_array(n=1,i=None,cmap="jet"):
    cm = plt.get_cmap(cmap)
    if cm is None:
        print "*** ERROR: colour_array(): colour map ",cmap," not found!"
        sys.exit(3)
    if n == 1:
        return "k"
    else:
        colarr = np.linspace(0.0,1.0,n)
        #colarr = np.arange(float(n))/float(n)
        colarr = cm(colarr)
        if i is not None:
            if i in range(n):
                colarr = colarr[i]
        return colarr


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_color(color):
    for hue in range(color):
        hue = 1. * hue / color
        col = [int(x) for x in colorsys.hsv_to_rgb(hue, 1.0, 230)]
        yield "#{0:02x}{1:02x}{2:02x}".format(*col)


def make_colourmap(seq):
    """
    make_colourmap(): Return a LinearSegmentedColormap.

    USAGE: cmap = make_colourmap(seq)

           seq: A sequence of floats and RGB-tuples. The floats should
                be increasing and be in the interval [0,1].

    e.g.  
    import matplotlib.colors as mcolors 
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('red'), c('violet'), c('blue')])
    rvb = make_colormap([c('red'), c('violet'), 0.33, c('violet'),\
                         c('blue'), 0.66, c('blue')])
    
    (From answer in
    http://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale)
    """
    import matplotlib.colors as mcolors
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)



####################################################################################
# LABEL FUNCTIONS
####################################################################################

def print_rounded_value(x,dx):
    return str(Decimal(str(x)).quantize(Decimal(str(dx))))

def frexp10(x):
    exp = int(math.log10(x)) 
    return x / 10**exp, exp

def sigfig(number,sigfig,latex=True):
    if sigfig == 0:
        return 0
    # Check if number is zero
    if number == 0:
        if sigfig == 1:
            return "0"
        else:
            return "0."+"0"*(sigfig-1)
    number = str(number)
    # Split mantissa and exponent
    if "e" in number.lower():
        m = number.split("e")[0]
        e = number.split("e")[1]
    else:
        m = number
        e = None
    # Check if negative
    neg = re.search('-',str(m))
    if neg is None:
        neg = False
    else:
        neg = True
        m = m.replace("-","")
    # Trim mantissa
    l = list(m)
    sigfigs = [i for i, x in enumerate(l) if fnmatch.fnmatch(x,'[1-9]')]
    if sigfig > len(sigfigs):
        lsf= sigfigs[-1]
    else:
        lsf= sigfigs[sigfig-1]
    if lsf < len(l)-1:
        nd = lsf+1
        if nd < len(l):
            if l[nd] == ".":
                nd += 1
            if int(l[nd]) >= 5:
                if int(l[lsf]) == 9:
                    pd = lsf
                    while int(l[pd])== 9:
                        l[pd] = "0"
                        pd -= 1
                        if l[pd] == '.':
                            pd -= 1
                    l[pd] = str(int(l[pd])+1)
                else:
                    l[lsf] = str(int(l[lsf])+1)
    if "." in l:
        dp = l.index(".")
        if dp > lsf:
            l = l[:dp]
        else:
            l = l[:lsf+1]
    m = "".join(l[:lsf+1])[::-1].zfill(len(l))[::-1]
    # Add minus if negative
    if neg:
        m ="-"+m
    # Add on exponent if present
    if e is None:
        number = m
    else:
        number = m+"e"+e
        if latex:
            number = number.replace("e","\\times 10^{")+"}"
    return number





####################################################################################
# LEGEND FUNCTIONS
####################################################################################

def Legend(ax,ec='none',fc='none',fontcolor="k",alpha=1.0,**kwargs):
    leg = ax.legend(**kwargs)
    frame = leg.get_frame()
    frame.set_edgecolor(ec)    
    frame.set_facecolor(fc)
    frame.set_alpha(alpha)
    for text in leg.get_texts():
        text.set_color(fontcolor)
    return


####################################################################################
# PLOTTING FUNCTIONS
####################################################################################


def ImageStats2D(ax,X,Y,Xbins,Ybins,Z=None,statistic="count",\
                           weights=None,func=None,**kwargs):
    from ..statistics.utils import binstats2D
    """                                                                                                                                                                                                                     
    statistic can be: mean,median,sum,product,std,var,percentile,avg,max,min,mode                                                                                                                                           
    NB 'avg' is weighted average                                                                                                                                                                                            
    """
    # Calculate statistic in 2-dimension bins                                                                                                                                                                               
    data,xedges,yedges,numb = binstats2D(X,Y,Xbins,Ybins=Ybins,Z=Z,\
                                             statistic=statistic,weights=weights)
    if func is not None:
        data = func(data)
    if np.any(np.isinf(data)):
        np.place(data,np.isinf(data),np.NaN)
    extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]
    # Set default preferences for selected keyword arguments                                                                                                                                                                
    if "extent" not in kwargs.keys():
        kwargs["extent"] = extent
    if "interpolation" not in kwargs.keys():
        kwargs["interpolation"] = "nearest"
    if "aspect" not in kwargs.keys():
        kwargs["aspect"] = "auto"
    if "origin" not in kwargs.keys():
        kwargs["origin"] = "lower"
    axim = ax.imshow(np.transpose(data),**kwargs)
    return axim
