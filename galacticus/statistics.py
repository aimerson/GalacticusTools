#! /usr/bin/env python

import numpy as np

def binstats(X,Y,Xbins,statistic="median",weights=None,mask=None):
    if weights is None:
        weights = np.ones_like(X)
    if mask is None:
        mask = np.ones(len(X),bool)
    statistic = str(statistic)
    try:
        p = float(statistic)
        def perc(x):
            return np.percentile(x,q=p)
        statistic = perc
    except ValueError:
        if statistic.lower().startswith("perc"):
            statistic = None
        elif statistic.lower() in ["average","avg"]:
            def avg(x):
                return np.average(x,weights=weights)
            statistic = avg
        elif statistic.lower() == "std":
            statistic = np.std
        elif statistic.lower() == "var":
            statistic = np.var
        elif statistic.lower().startswith("prod"):
            statistic = np.prod
        elif statistic.lower().startswith("min"):
            statistic = np.nanmin
        elif statistic.lower().startswith("max"):
            statistic = np.nanmax
        elif statistic.lower() == "mode":
            def mode_of_data(x):
                return mode(x)[0][0]
            statistic = mode_of_data
        else:
            pass
    if statistic is None:
        result,bin_edges,binnumber = binned_statistic(X[mask],Y[mask],statistic="count",bins=Xbins)
        result = result*100.0/float(np.sum(result))
    else:
        result,bin_edges,binnumber = binned_statistic(X[mask],Y[mask],statistic=statistic,bins=Xbins)
    return result,bin_edges,binnumber



def binstats2D(X,Y,Xbins,Ybins=None,Z=None,statistic="median",weights=None):
    """                                                                                                                                                                                                                     
    statistic can be: mean,median,sum,product,std,var,percentile,avg,max,min,mode,fraction,percentage                                                                                                                       
                                                                                                                                                                                                                            
    NB 'avg' is weighted average                                                                                                                                                                                            
    """
    if statistic is None:
        statistic = "count"
    if Ybins is None:
        Ybins = np.copy(Xbins)
    bins = [Xbins,Ybins]
    if Z is None:
        Z = np.ones_like(X)
    statistic = str(statistic)
    try:
        p = float(statistic)
        def perc(x):
            return np.percentile(x,q=p)
        statistic = perc
    except ValueError:
        if statistic.lower().startswith("perc"):
            pass
        elif statistic.lower().startswith("frac"):
            pass
        elif statistic.lower() in ["average","avg"]:
            if weights is None:
                weights = np.ones_like(X)
            def avg(x):
                return np.average(x,weights=weights)
            statistic = avg
        elif statistic.lower() == "std":
            statistic = np.std
        elif statistic.lower() == "var":
            statistic = np.var
        elif statistic.lower().startswith("prod"):
            statistic = np.prod
        elif statistic.lower().startswith("min"):
            statistic = np.nanmin
        elif statistic.lower().startswith("max"):
            statistic = np.nanmax
        elif statistic.lower() == "mode":
            def mode_of_data(x):
                return mode(x)[0][0]
            statistic = mode_of_data
        else:
            pass
    if statistic.lower() == "count":
        stat,xedges,yedges,numb = binned_statistic_2d(X,Y,Z,statistic="count",bins=bins)
        if weights is not None:
            stat,xedges,yedges = np.histogram2d(X,Y,bins=bins,weights=weights)
    elif any([statistic.lower().startswith("perc"),statistic.lower().startswith("frac")]):
        stat,xedges,yedges,numb = binned_statistic_2d(X,Y,Z,statistic="count",bins=bins)
        stat = stat/float(np.sum(stat))
        if statistic.lower().startswith("perc"):
            stat = stat*100.0
    else:
        stat,xedges,yedges,numb = binned_statistic_2d(X,Y,Z,statistic=statistic,bins=bins)
    stat = Inf2NaN2D(stat)
    return stat,xedges,yedges,numb
