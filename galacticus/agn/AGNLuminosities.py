#! /usr/bin/env python

import sys,re
from .agnSpectra import agnSpectralTables
from ..Luminosities import LuminosityClass


def parseAGNLuminosity(datasetName):
    funcname = sys._getframe().f_code.co_name
    # Extract dataset name information
    searchString = "^agnLuminosity:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
        "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<absorption>:noAbsorption)?"+\
        "(?P<alphaString>:alpha(?P<alpha>[0-9\-\+\.]+))?$"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class AGNLuminosityClass(LuminosityClass):

    def __init__(self,datasetName=None,luminosity=None,\
                     redshift=None,outputName=None):
        super(AGNLuminosityClass,self).__init__(datasetName=datasetName,luminosity=luminosity,\
                                                    redshift=redshift,outputName=outputName)
        return
