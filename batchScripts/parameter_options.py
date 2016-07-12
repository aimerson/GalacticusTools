#! /usr/bin/env python

import sys,os,fnmatch
import re
import subprocess
from galacticus.parameters import GalacticusParameters
from galacticus.EmissionLines import getLineNames




def select_parameters(paramfile,jobArrayID=-1,jobArraySize=1):

    reffile = "parameters_ref.xml"    
    GP = GalacticusParameters(reffile)    
    params = []
    #params.append(("verbosityLevel",2))


    if fnmatch.fnmatch(paramfile.lower(),"emline_mc*.xml"):
        SDSS = "u g r i z".split()
        UKIRT = "J H K".split()
        Cont = "Lyc HeliumContinuum OxygenContinuum".split()        
        filters = [ "SDSS_"+f for f in SDSS ] + [ "UKIRT_"+f for f in UKIRT ] +  [ f for f in Cont ] 
        #filter = filters + ["topHatArray_5000_6000_100"]
        #resolutions = [100,250,500,750,1000]
        resolutions = [100]
        for line in getLineNames():
            filters = filters + ["emissionLineContinuumPair_"+line+"_"+str(i) for i in resolutions]            
            filters = filters + ["emissionLineContinuumCentral_"+line+"_"+str(i) for i in resolutions]
        filters_type = ["rest"]*len(filters) +  ["observed"]*len(filters)        
        filters = filters + filters        
        filters_postprocess = ["default", "recent"]
        params.append(("luminosityFilter",filters))
        params.append(("luminosityRedshift",["all"]*len(filters)))
        params.append(("luminosityType",filters_type))
        params.append(("spheroidOutputStarFormationRate","true"))
        params.append(("diskOutputStarFormationRate","true"))
        params.append(("outputRedshifts",[0.0]))        
        params.append(("mergerTreeBuildTreesPerDecade",re.sub("[^0-9]", "",paramfile)))
        #params.append(("outputHalfLightData","true"))
        params.append(("outputHalfMassData","true"))

        params.append(("outputDensityContrastData","true"))
        params.append(("outputDensityContrastValues",[200,500,1500])) # relative to background
        params.append(("outputDensityContrastDataDarkOnly","false")) # 200,500,etc. relative to DM only or all matter
        params.append(("outputDensityContrastHaloLoaded","true")) # include adiabatic contraction?        
        params.append(("outputVirialData","true")) # includes concentration, virial radius and velocity
        params.append(("hotHaloOutputCooling","true"))
        

    if fnmatch.fnmatch(paramfile.lower(),"hothalo_mc*.xml"):
        SDSS = "u g r i z".split()
        UKIRT = "J H K".split()
        Cont = "Lyc HeliumContinuum OxygenContinuum".split()        
        filters = [ "SDSS_"+f for f in SDSS ] + [ "UKIRT_"+f for f in UKIRT ] +  [ f for f in Cont ] 
        filters = filters + ["xRayFull","xRaySoft","xRayHard","Galex_FUV","Galex_NUV"]
        filters_type = ["rest"]*len(filters) +  ["observed"]*len(filters)        
        params.append(("luminosityFilter",filters+filters))
        params.append(("luminosityRedshift",["all"]*len(filters)+["all"]*len(filters)))
        params.append(("luminosityType",filters_type))
        params.append(("spheroidOutputStarFormationRate","true"))
        params.append(("diskOutputStarFormationRate","true"))
        params.append(("outputRedshifts",[0.0]))        

        params.append(("hotHaloOutputCooling","true"))
        #params.append(("outputHalfLightData","true"))
        params.append(("outputHalfMassData","true"))

        params.append(("mergerTreeBuildHaloMassMinimum",1.0e12))
        params.append(("mergerTreeBuildHaloMassMaximum",2.0e15))
        params.append(("haloMassFunctionSamplingAbundanceMinimum",1.0e-3))
        params.append(("haloMassFunctionSamplingAbundanceMaximum",1.0e-6))
        params.append(("mergerTreeBuildTreesPerDecade",re.sub("[^0-9]", "",paramfile)))

        params.append(("outputDensityContrastData","true"))
        params.append(("outputDensityContrastValues",[200,500,1500])) # relative to background
        params.append(("outputDensityContrastDarkOnly","false")) # 200,500,etc. relative to DM only or all matter
        params.append(("outputDensityContrastHaloLoaded","true")) # include adiabatic contraction?        
        params.append(("outputVirialData","true")) # includes concentration, virial radius and velocity

        
        params.append(("blackHoleOutputAccretion","true"))


        if jobArrayID>0:
            params.append(("treeEvolveWorkerCount",jobArraySize))
            params.append(("treeEvolveWorkerNumber",jobArrayID))

        # haloMassFunctionSamplingMethod

    if paramfile == "nbody_trial.xml":
        SDSS = "u g r i z".split()
        UKIRT = "J H K".split()
        Cont = "Lyc HeliumContinuum OxygenContinuum".split()        
        filters = [ "SDSS_"+f for f in SDSS ] + [ "UKIRT_"+f for f in UKIRT ] +  [ f for f in Cont ] 
        resolutions = [100,250,500,750,1000]
        for line in getLineNames():
            filters = filters + ["emissionLineContinuumPair_"+line+"_"+str(i) for i in resolutions]            
            filters = filters + ["emissionLineContinuumCentral_"+line+"_"+str(i) for i in resolutions]
        
        filters_type = ["rest"]*len(filters) +  ["observed"]*len(filters)                
        filters = filters + filters
        #filters_postprocess = ["default"]*len(filters) + ["recent"]*len(filters)
        #filters = filters + filters
        #filters_type = filters_type + filters_type
        
        params.append(("luminosityFilter",filters))
        params.append(("luminosityRedshift",["all"]*len(filters)))
        params.append(("luminosityType",filters_type))
        #params.append(("luminosityPostprocessSet",filters_postprocess))
        #params.append(("stellarPopulationSpectraPostprocessRecentMethods","inoue2014 recent"))
        #params.append(("stellarPopulationSpectraRecentTimeLimit",1.0e-2))


        params.append(("spheroidOutputStarFormationRate","true"))
        params.append(("diskOutputStarFormationRate","true"))
        params.append(("treeNodeMethodSatellite","preset"))
        params.append(("treeNodeMethodPosition","preset"))
        params.append(("mergerTreeConstructMethod","read"))
        params.append(("mergerTreeReadFileName","/nobackup0/sunglass/amerson/simulations/millennium/milliMillennium/milliMillennium.hdf5"))
        params.append(("allTreesExistAtFinalTime","false"))

        params.append(("cosmologyParametersMethod","simple"))
        params.append(("HubbleConstant",73.0,"cosmologyParametersMethod"))
        params.append(("OmegaMatter",0.25,"cosmologyParametersMethod"))
        params.append(("OmegaDarkEnergy",0.75,"cosmologyParametersMethod"))
        params.append(("OmegaBaryon",0.0455,"cosmologyParametersMethod"))        
        params.append(("sigma_8",0.9,"cosmologicalMassVarianceMethod"))        

        params.append(("outputDensityContrastData","true"))
        params.append(("outputDensityContrastValues",[200,500])) # relative to background
        params.append(("outputDensityContrastDarkOnly","false")) # 200,500 relative to DM only or all matter
        params.append(("outputDensityContrastHaloLoaded","true")) # include adiabatic contraction?        
        params.append(("outputVirialData","true")) # includes concentration, virial radius and velocity

        params.append(("powerSpectrumPrimordialMethod","powerLaw"))        
        params.append(("index",0.961,"powerSpectrumPrimordialMethod"))
        params.append(("wavenumberReference",1,"powerSpectrumPrimordialMethod"))
        params.append(("running",0,"powerSpectrumPrimordialMethod"))

        params.append(("metaCollectTimingData","true"))

        #params.append(("mergerTreeReadPresetScaleRadii","true"))
        #params.append(("mergerTreeReadPresetScaleRadiiFailureIsFatal","true"))
        #params.append(("mergerTreeReadPresetScaleRadiiConcentrationMinimum",3))
        #params.append(("mergerTreeReadPresetScaleRadiiConcentrationMaximum",60))
        #params.append(("mergerTreeReadPresetScaleRadiiMinimumMass","see below"))

        params.append(("mergerTreeReadPresetScaleRadiiMinimumMass",2.5e11))
        
        params.append(("virialDensityContrastMethod","percolation"))
        params.append(("virialDensityContrastPercolationLinkingLength","0.2"))
        
        
        #params.append(("outputHalfLightData","true"))
        #params.append(("outputHalfMassData","true"))

        params.append(("mergerTreeReadPresetScaleRadii","false"))


    # Set global job array options
    if jobArrayID>0:
        params.append(("galacticusOutputFileName","galacticus_"+str(jobArrayID)+".hdf5"))
        paramfile = paramfile.replace(".xml","_"+str(jobArrayID)+".xml")

    for t in params:
        if len(t) == 2:
            parent = None
        else:
            parent = t[2]
        GP.set_parameter(t[0],t[1],parent=parent)            
    GP.write(paramfile)

    return paramfile
    



    


