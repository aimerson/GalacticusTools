#! /usr/bin/env python

import sys
sys.path.append("/home/amerson/codes/Galacticus/galacticustools/")
from galacticus.parameters import GalacticusParameters


def select_parameters(paramfile):

    reffile = "parameters_ref.xml"    
    GP = GalacticusParameters(reffile)    
    params = {}

    if paramfile == "trial_run.xml":
        SDSS = "u g r i z".split()
        UKIRT = "J H K".split()
        Cont = "Lyc HeliumContinuum OxygenContinuum".split()        
        filters = [ "SDSS_"+f for f in SDSS ] + [ "UKIRT_"+f for f in UKIRT ] +  [ f for f in Cont ] 
        filters_type = ["rest"]*len(filters) +  ["observed"]*len(filters)        
        params["luminosityFilter"] = filters + filters
        params["luminosityRedshift"] = ["all"]*len(filters) + ["all"]*len(filters)
        params["luminosityType"] = filters_type
        params["spheroidOutputStarFormationRate"] = "true"
        params["diskOutputStarFormationRate"] = "true"
        params["outputRedshifts"] = [0.0,0.5,1.0]
    
    for p in params:
        GP.set_parameter(p,params[p])
    GP.write(paramfile)
    return
    


    


