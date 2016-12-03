#! /usr/bin/env python

import sys,os,glob,fnmatch
import numpy as np
from .utils import *
from ..constants import Pi
from ..GalacticusErrors import ParseError

from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import  mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
import matplotlib.ticker 
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter


class angularGeometry(object):
    
    def __init__(self,units="degrees"):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if fnmatch.fnmatch(units.lower(),"deg*"):
            self.units = "degrees"
            self.latex = '$^{\circ}$'
            self.wrapLimit = 360.0
        elif fnmatch.fnmatch(units.lower(),"rad*"):
            self.units = "radians"
            self.latex = '$^{\\rm r}$'
            self.wrapLimit = 2.0*Pi
        elif fnmatch.fnmatch(units.lower(),"h*r*"):
            self.units = "hours"
            self.latex = '$^{\\rm h}$'
            self.wrapLimit = 24.0
        else:
            err = classname+"(): angular units not recognised! Units = '"+units+"'"+\
                "\n       Options: degrees (default), radians, hours"
            raise ValueError(err)        
        return

    def wrap(self,minAngle,maxAngle):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name               
        if minAngle < 0:
            minAngle = self.wrapLimit + minAngle
        if maxAngle < 0:
            maxAngle = self.wrapLimit + maxAngle
        if maxAngle < minAngle:
            maxAngle = maxAngle + self.wrapLimit                
        return minAngle,maxAngle

    def fullSky(self,minAngle,maxAngle):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name               
        return minAngle==0.0 and maxAngle==self.wrapLimit

    def getTicks(self,minAngle,maxAngle,diffAngle,sigFig=2):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name               
        minAngle,maxAngle = self.wrap(minAngle,maxAngle)
        tickValues = np.arange(minAngle,np.minimum(maxAngle,self.wrapLimit),diffAngle)
        tickLabels = np.copy(tickValues)
        mask = tickLabels >= self.wrapLimit
        np.place(tickLabels,mask,tickLabels[mask]-self.wrapLimit)
        tickLabels = [sigfig(label,sigFig)+self.latex for label in tickLabels]
        return list(zip(list(self.convert(tickValues,units='deg')),tickLabels))

    def convert(self,angle,units='degrees'):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name               
        if fnmatch.fnmatch(units.lower(),"deg*"):
            units = "degrees"            
        elif fnmatch.fnmatch(units.lower(),"rad*"):
            units = "radians"
        elif fnmatch.fnmatch(units.lower(),"h*r*"):
            units = "hours"
        else:
            err = funcname+"(): angular units not recognised! Units = '"+units+"'"+\
                "\n       Options: degrees (default), radians, hours"
            raise ValueError(err)
        factor = 1.0
        if units == "degrees":
            if self.units == "hours":
                factor = 15.0
            if self.units == "radians":
                factor = 180.0/Pi
        if units == "hours":
            if self.units == "degrees":
                factor = 1.0/15.0
            if self.units == "radians":
                factor = 180.0/Pi/15.0
        if units == "radians":
            if self.units == "hours":
                factor = (Pi/180.0)*15.0
            if self.units == "degrees":
                factor = Pi/180.0            
        return angle*factor

        
class radialGeometry(object):
    
    def __init__(self,units=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.redshift = False
        if units is None:
            self.units = None
            self.redshift = True
            self.latex = ""
        elif fnmatch.fnmatch(units.lower(),"[mkg]pc"):
            self.units = units[0].upper()+"pc"
            self.latex = '$\,\mathrm{'+self.units+'}$'
        elif fnmatch.fnmatch(units.lower(),"[mkg]pc/h"):
            self.units = units[0].upper()+"pc"
            self.latex = '$\,h^{-1}\,\mathrm{'+self.units+'}$'
        elif fnmatch.fnmatch(units.lower(),"*yr"):
            self.units = units[0].upper()+"yr"
            self.latex = '$\,\mathrm{'+self.units+'}$'            
        else:
            err = classname+"(): units not recognised! Units = '"+units+"'." + \
                "\n      Options: None (redshift, default), Mpc, Mpc/h, Gpc, Gpc/h, kpc, kpc/h, Lyr, yr, Gyr" 
            raise ValueError(err)
        return

    def getTicks(self,minDistance,maxDistance,diffDistance,sigFig=2):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        tickValues = list(np.arange(minDistance,maxDistance,diffDistance))
        tickLabels = [sigfig(value,sigFig)+self.latex for value in tickValues]
        return list(zip(list(tickValues),tickLabels))




class conePlot(object):
    
    def __init__(self,fig,angularRange,radialRange,\
                     angularUnits="degrees",angularSigFig=2,angularLabel=None,\
                     radialUnits=None,radialSigFig=2,radialLabel=None,\
                     subplot=111,rotate=0.0,gridlines=True,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Store figure
        self.fig = fig
        # Set angular range
        self.ANGULAR = angularGeometry(units=angularUnits)
        angularTicks = self.ANGULAR.getTicks(angularRange[0],angularRange[1],angularRange[2],sigFig=angularSigFig)
        # Set grid lines for angular coordinates        
        angular_grid_locator = FixedLocator([v for v, s in angularTicks])
        angular_tick_formatter = DictFormatter(dict(angularTicks))
        # Set radial range
        self.RADIAL = radialGeometry(units=radialUnits)
        radialTicks = self.RADIAL.getTicks(radialRange[0],radialRange[1],radialRange[2],sigFig=radialSigFig)
        # Set radial grid lines
        radial_grid_locator = FixedLocator([v for v, s in radialTicks])
        # Check whether creating special case of all-sky axes
        self.fullSky = self.ANGULAR.fullSky(angularRange[0],angularRange[1])
        if self.fullSky:
            # Create axes 
            self.ax = self.fig.add_subplot(subplot,polar=True)
            # Set radial tickmarks
            self.ax.set_ylim(0.0,radialRange[1])
            tick_labels = np.arange(0.0,radialRange[1],radialRange[2])[1:]
            majorLocator = matplotlib.ticker.FixedLocator(tick_labels)
            self.ax.yaxis.set_major_locator(majorLocator)
            # Set angular tick marks
            self.ax.set_xlim(0.0,360.0)
            angularDiff = self.ANGULAR.convert(angularRange[2],units='radians')
            tick_labels = np.arange(0.0,(2.0*Pi)+angularDiff,angularDiff)
            majorLocator = matplotlib.ticker.FixedLocator(tick_labels)
            self.ax.xaxis.set_major_locator(majorLocator)
            labels = [s for v, s in angularTicks]
            if angularSigFig <= 2:
                labels[0] = labels[0].replace(".0","")
            self.ax.set_xticklabels(labels)
            # Show axis labels?
            if angularLabel is not None:
                self.ax.set_xlabel(angularLabel)
            # Show gridlines?
            self.ax.grid(gridlines,ls='-',c='grey',alpha=0.25)
            self.aux_ax = None
            self.radialAxis = self.ax.yaxis
            self.angularAxis = self.ax.xaxis
            return
        # Set rotation/orientation of axes such that wedge is
        # symmetrical about +/-X or +/-Y axis
        minAngle,maxAngle = self.ANGULAR.wrap(angularRange[0],angularRange[1])
        minAngleDeg = self.ANGULAR.convert(minAngle,units='deg')
        maxAngleDeg = self.ANGULAR.convert(maxAngle,units='deg')
        midway = minAngleDeg + (maxAngleDeg-minAngleDeg)/2.0
        if rotate < 0.0:
            rotate = 360.0 + rotate
        tr_rotate = Affine2D().translate(rotate-midway, 0)
        # Scale degree to radians
        tr_scale = Affine2D().scale(np.pi/180., 1.)
        # Apply rotation and scaling to polar axes
        tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()
        # Construct grid, setting angular and radial limits        
        extremes = (minAngleDeg,maxAngleDeg,radialRange[0],radialRange[1])
        grid_helper = floating_axes.GridHelperCurveLinear(tr,extremes=extremes,\
                                                              grid_locator1=angular_grid_locator,\
                                                              grid_locator2=radial_grid_locator,\
                                                              tick_formatter1=angular_tick_formatter,\
                                                              tick_formatter2=None)
        # Construct axes
        self.ax = floating_axes.FloatingSubplot(self.fig,subplot,grid_helper=grid_helper)
        self.fig.add_subplot(self.ax)
        # Adjust angular tick marks and labels according to orientation
        self.angularAxis = self.ax.axis["top"]
        if(radialRange[0] == 0.0):
            self.ax.axis["bottom"].set_visible(False)
        else:
            self.ax.axis["bottom"].set_axis_direction("top")
            self.ax.axis["bottom"].toggle(ticklabels=False,label=False)
        self.angularAxis.set_axis_direction("bottom")
        self.angularAxis.toggle(ticklabels=True,label=True)
        if rotate <= 180.0:
            self.angularAxis.major_ticklabels.set_axis_direction("top")
            self.angularAxis.label.set_axis_direction("top")
        else:
            self.angularAxis.major_ticklabels.set_axis_direction("bottom")
            self.angularAxis.label.set_axis_direction("bottom")

        # Adjust radial tick marks and labels according to orientation
        if rotate <= 90.0 or rotate >= 270.0:
            self.radialAxis1 = self.ax.axis["left"]
            self.radialAxis2 = self.ax.axis["right"]
            self.radialAxis1.set_axis_direction("bottom")
            self.radialAxis2.set_axis_direction("top")
        else:
            self.radialAxis1 = self.ax.axis["right"]
            self.radialAxis2 = self.ax.axis["left"]
            self.radialAxis1.set_axis_direction("top")
            self.radialAxis2.set_axis_direction("bottom")
        self.radialAxis1.toggle(ticklabels=True, label=True)
        self.radialAxis2.toggle(ticklabels=True, label=True)                    
        if rotate > 45.0 and rotate < 135.0:
            self.radialAxis1.major_ticklabels.set_axis_direction("right")
            self.radialAxis1.label.set_axis_direction("right")
            self.radialAxis2.major_ticklabels.set_axis_direction("left")
        if rotate >= 135.0 and rotate <= 225.0:
            self.radialAxis1.major_ticklabels.set_axis_direction("bottom")
            self.radialAxis1.label.set_axis_direction("bottom")
            self.radialAxis2.major_ticklabels.set_axis_direction("top")
        if rotate > 225.0 and rotate < 315.0:
            self.radialAxis1.major_ticklabels.set_axis_direction("left")
            self.radialAxis1.label.set_axis_direction("left")
            self.radialAxis2.major_ticklabels.set_axis_direction("right")
        if radialLabel is not None:                                                                                                                                                                                    
            self.radialAxis1.label.set_text(radialLabel+"["+self.RADIAL.latex+"]")
            self.radialAxis1.label.set_size(20.0)        
        # Add gridlines to plot? (Default = True)
        self.ax.grid(gridlines)
        # Create a parasite axes whose transData in RA, z
        self.aux_ax = self.ax.get_aux_axes(tr)        
        # For aux_ax to have a clip path as in ax but this has a side
        # effect that the patch is drawn twice, and possibly over some
        # other artists. So, we decrease the zorder a bit to prevent this.
        self.aux_ax.patch = self.ax.patch
        self.ax.patch.zorder=0.9
        return


    
