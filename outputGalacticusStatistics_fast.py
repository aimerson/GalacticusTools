#! /usr/bin/env python

import glob,fnmatch,re
import numpy as np
from galacticus.hdf5 import HDF5
from galacticus.io import GalacticusHDF5
from galacticus.Luminosities import ergPerSecond
from galacticus.constants import Pi
from galacticus.utils.timing import STOPWATCH
from galacticus.utils.progress import Progress
from galacticus.cosmology import Cosmology,MpcToCM
from galacticus.dust.utils import AtoTau,TauToA
from galacticus.dust.screens import screenModel
from galacticus.satellites import getHostIndex
from galacticus.statistics.hods import HaloOccupationDistribution


DUST = screenModel("calzetti")


class statisticsFile(object):
    
    def __init__(self,outdir,ofile,massBins=np.arange(9.6,15.0,0.2),lumBins=np.arange(37.0,44.0,0.1)):        
        self.outdir = outdir
        self.ofile = ofile
        self.OUT = HDF5(ofile,'w')
        self.COSMOLOGY = Cosmology(omega0=0.25,lambda0=0.75,h0=0.73,omegab=0.045,h_independent=True)        
        # Store mass and luminosity bins
        self.massBins = massBins
        self.lumBins = lumBins
        # Create HOD class
        self.HOD = HaloOccupationDistribution(self.massBins)
        self.OUT.addDataset("/","M200binCentres",self.HOD.binCentres)
        self.OUT.addAttributes("/M200binCentres",{"units":"Msol/h"})        
        self.luminosityLimits = "1.0e38,1.0e39,1.0e40,1.0e41,1.0e42,3.0e42,5.0e42,7.0e42,9.0e42,1.0e43".split(",")
        self.fluxLimits = "1e-17,5e-17,1e-16,2e-16,3e-16".split(",")
        # Store luminosity function bins
        dl = self.lumBins[1] - self.lumBins[0]
        bins = self.lumBins[:-1] + dl/2.0
        self.OUT.addDataset("/","LuminositybinCentres",bins)
        self.OUT.addAttributes("/LuminositybinCentres",{"units":"h^{-2} erg/s"})
        # Vectors to store galaxy properties
        self.dL = None
        self.z = None
        self.luminosity = None
        self.obsluminosity = None
        self.nii = None
        self.central = None
        self.satellite = None
        self.weights = None
        return

    def reset(self):
        self.dL = None
        self.z = None
        self.luminosity = None
        self.obsluminosity = None
        self.nii = None
        self.central = None
        self.satellite = None
        self.weights = None
        return

    def getCalzettiAttenuation(self,z):
        atten = -0.0652*z + 1.46
        return atten

    def processHOD(self,outputNumber,outputName,PROG=None):        
        # Extract info from output name        
        searchString = "^(?P<blended>Blended)?(?P<luminosity>Luminosity|Flux)"+\
            "_(?P<limit>[-+]?[0-9]*\.?[0-9]+[eE][-+]?[0-9]+)"+\
            "_dust(?P<dust>Free|Attenuated)$"
        MATCH = re.search(searchString,outputName)        
        # [NII] contribution?
        if MATCH.group('blended') is None:
            niiRatio = np.ones_like(self.luminosity)
        else:
            niiRatio = np.copy(self.nii)
        # Luminosity or flux?
        if MATCH.group('luminosity') == "Flux":
            luminosity = np.copy(self.obsluminosity)/(4.0*Pi*(self.dL**2))
        else:
            luminosity = np.copy(self.luminosity)
        # Dust attenuation
        if MATCH.group('dust') == "Free":
            atten = 1.0
        else:
            atten = self.getCalzettiAttenuation(self.z[0])
        # Get limit
        limit = float(MATCH.group('limit'))
        # Create mask
        mask = luminosity*atten*niiRatio >= limit
        # Compute HOD
        self.HOD.resetHOD()
        if any(mask):
            self.HOD.addHalosToHOD(mask=mask,verbose=False)
        hod = self.HOD.computeHOD().view(np.recarray)                                
        # Store information to HDF5 file        
        name = outputName
        self.OUT.addDataset(outputNumber+"/HOD","hod"+name,np.copy(hod.allGalaxies))
        self.OUT.addDataset(outputNumber+"/HOD","hod"+name+"_err",np.copy(hod.allGalaxiesError))                    
        self.OUT.addDataset(outputNumber+"/HOD","hodCentrals"+name,np.copy(hod.centrals))
        self.OUT.addDataset(outputNumber+"/HOD","hodCentrals"+name+"_err",np.copy(hod.centralsError))            
        self.OUT.addDataset(outputNumber+"/HOD","hodSatellites"+name,np.copy(hod.satellites))
        self.OUT.addDataset(outputNumber+"/HOD","hodSatellites"+name+"_err",np.copy(hod.satellitesError))            
        if PROG is not None:
            PROG.increment()
            PROG.print_status_line()
        return

    def processLF(self,outputNumber,outputName,PROG=None):        
        # Extract info from output name        
        searchString = "^(?P<blended>Blended)?Dust(?P<dust>Free|Attenuated)$"
        MATCH = re.search(searchString,outputName)        
        # [NII] contribution?
        if MATCH.group('blended') is None:
            niiRatio = np.ones_like(self.luminosity)
        else:
            niiRatio = np.copy(self.nii)
        # Dust attenuation
        if MATCH.group('dust') == "Free":
            atten = 1.0
        else:
            atten = self.getCalzettiAttenuation(self.z[0])
        # Compute luminosity            
        dl = self.lumBins[1] - self.lumBins[0]
        logL = np.log10(self.luminosity*atten*niiRatio+1.0e-50)
        lf,bins = np.histogram(logL,self.lumBins,weights=self.weights)    
        lf = lf.astype(float)
        lf /= dl
        name = outputName
        self.OUT.addDataset(outputNumber+"/LF","lf"+name,np.copy(lf))
        lf,bins = np.histogram(logL[self.central],self.lumBins,weights=self.weights[self.central])    
        lf = lf.astype(float)
        lf /= dl
        self.OUT.addDataset(outputNumber+"/LF","lfCentrals_"+name,np.copy(lf))
        lf,bins = np.histogram(logL[self.satellite],self.lumBins,weights=self.weights[self.satellite])
        lf = lf.astype(float)
        lf /= dl
        self.OUT.addDataset(outputNumber+"/LF","lfSatellites_"+name,np.copy(lf))            
        if PROG is not None:
            PROG.increment()
            PROG.print_status_line()
        return

    def processOutput(self,iout):
        self.reset()                
        # Open file and read redshift
        outputNumber = str(iout+1).zfill(3)
        galaxiesFile = self.outdir + "/galacticusGalaxiesOutput"+outputNumber+".npy"
        print("Galaxies file = "+galaxiesFile)
        galaxies = np.load(galaxiesFile).view(np.recarray)
        self.z = np.copy(galaxies["snapshotRedshift"])    
        print("Redshift = "+str(self.z[0]))
        self.dL = MpcToCM(self.COSMOLOGY.luminosity_distance(self.z))        
        # Create HDF5 data structure
        self.OUT.mkGroup(outputNumber)
        self.OUT.addAttributes(outputNumber,{"redshift":self.z[0]})
        self.OUT.mkGroup(outputNumber+"/HOD")        
        self.OUT.mkGroup(outputNumber+"/LF")
        self.HOD.reset()
        self.HOD.addHalos(galaxies,verbose=True,weightHalos=False)
        # Store galaxy properties
        emlines = fnmatch.filter(galaxies.dtype.names,"totalLineLuminosity:balmerAlpha6563:rest*")
        self.luminosity = np.copy(galaxies[emlines[0]]) + 1.0e-50
        emlines = fnmatch.filter(galaxies.dtype.names,"*LineLuminosity:balmerAlpha6563:observed*")
        self.obsluminosity = np.copy(galaxies[emlines[0]]) + 1.0e-50
        self.nii = np.copy(galaxies["niiHalphaRatio"])
        np.place(self.nii,np.isnan(self.nii),0.0)
        self.central = galaxies["nodeIsIsolated"] == 1
        self.satellite = galaxies["nodeIsIsolated"] == 1
        self.weights = galaxies["weight"]
        # Compute HODs (luminosity selected)
        print("Processing luminosity-limited HODs..")        
        outputNames = ["Luminosity_"+str(limit)+"_dustFree" for limit in self.luminosityLimits] + \
            ["BlendedLuminosity_"+str(limit)+"_dustFree" for limit in self.luminosityLimits] + \
            ["Luminosity_"+str(limit)+"_dustAttenuated" for limit in self.luminosityLimits] + \
            ["BlendedLuminosity_"+str(limit)+"_dustAttenuated" for limit in self.luminosityLimits]
        PROG = Progress(len(outputNames))
        dummy = [self.processHOD(outputNumber,name,PROG=PROG) for name in outputNames]
        del dummy
        # Compute HODs (flux selected)
        print("Processing flux-limited HODs..")
        outputNames = ["Flux_"+str(limit)+"_dustFree" for limit in self.fluxLimits] + \
            ["BlendedFlux_"+str(limit)+"_dustFree" for limit in self.fluxLimits] + \
            ["Flux_"+str(limit)+"_dustAttenuated" for limit in self.fluxLimits] + \
            ["BlendedFlux_"+str(limit)+"_dustAttenuated" for limit in self.fluxLimits]
        PROG = Progress(len(outputNames))
        dummy = [self.processHOD(outputNumber,name,PROG=PROG) for name in outputNames]
        del dummy
        # Compute luminosity functions
        print("Processing luminosity functions...")
        outputNames = ["DustFree","DustAttenuated","BlendedDustFree","BlendedDustAttenuated"]
        PROG = Progress(len(outputNames))
        dummy = [self.processLF(outputNumber,name,PROG=PROG) for name in outputNames]
        return

        
        

makeFile = True
statsFile = "/home/amerson/Projects/Galacticus/HalphaHODs/galacticusStatistics_fast.hdf5"

if makeFile:
    WATCH = STOPWATCH()
    print("Building file: " +statsFile+"...")
    outdir = "/halo_nobackup/sunglass/amerson/Galacticus_Out/v0.9.4/halphaHOD/20180301/"    
    massBins = np.arange(9.6,15.0,0.2)
    lumBins = np.arange(37.0,44.0,0.1)
    STATS = statisticsFile(outdir,statsFile,massBins=massBins,lumBins=lumBins)
    outputs = np.arange(31,dtype=int)
    #outputs = np.arange(5,dtype=int)
    dummy = [STATS.processOutput(iout) for iout in outputs]
    STATS.OUT.close()    
    print("HDF5 file written")
    WATCH.stop()
