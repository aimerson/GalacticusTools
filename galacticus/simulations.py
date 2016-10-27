#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
import pkg_resources
import xml.etree.ElementTree as ET
from .cosmology import Cosmology

class Simulation(object):
    
    def __init__(self,simulation,verbose=False,radiation=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.verbose = verbose
        
        # Load xml file of simulation specifications
        self.xmlFile = locate_simulation_file(simulation)
        xmlStruct = ET.parse(self.xmlFile)
        xmlRoot = xmlStruct.getroot()
        self.name = xmlRoot.attrib["name"]
        xmlMap = {c.tag:p for p in xmlRoot.iter() for c in p}
        self.boxSize = xmlRoot.find("boxSize").text
        self.boxSizeUnits = xmlRoot.find("boxSize").attrib["units"]
        particles = xmlRoot.find("particles")
        self.particleMass = particles.find("mass").text
        self.particleMassUnits = particles.find("mass").attrib["units"]
        self.particleNumber = particles.find("number").text
        snapshots = xmlRoot.find("snapshots")
        snapshotData = snapshots.findall("snapshot")
        self.snapshots = np.zeros(len(snapshotData),dtype=[("index",int),("z",float)])
        for i,snap in enumerate(snapshotData):
            self.snapshots["z"][i] = float(snap.text)
            self.snapshots["index"][i] = int(snap.attrib["number"])
        self.snapshots = self.snapshots.view(np.recarray)
        cosmologyStruct = xmlRoot.find("cosmology")
        self.omega0 = float(cosmologyStruct.find("OmegaM").text)
        self.lambda0 = float(cosmologyStruct.find("OmegaL").text)
        self.omegaB = float(cosmologyStruct.find("OmegaB").text)
        self.h0 = float(cosmologyStruct.find("H0").text)/100.0
        self.sigma8 = float(cosmologyStruct.find("sigma8").text)
        self.ns = float(cosmologyStruct.find("ns").text)
        try:
            self.temperatureCMB = float(cosmologyStruct.find("temperatureCMB").text)
        except AttributeError:
            self.temperatureCMB = 2.726
        self.cosmology = Cosmology(omega0=self.omega0,lambda0=self.lambda0,omegab=self.omegaB,h0=self.h0,\
                                       sigma8=self.sigma8,ns=self.ns,radiation=radiation,\
                                       zmax=self.snapshots.z.max(),nzmax=10000)               
        if self.verbose:
            print("------------------------------------------------------"                           )
            print(" SPECIFICATIONS: "+self.name                                                      )
            print("            BOX SIZE        = "+str(self.boxSize)+" "+self.boxSizeUnits           )
            print("            NUM. PARTICLES  = "+str(self.particleNumber)                          )
            print("            PARTICLE MASS   = "+str(self.particleMass)+ " "+self.particleMassUnits)
            print("            MIN. REDSHIFT   = "+str(self.snapshots.z.min())                       )
            print("       Cosmology:"                                                                )
            print("            OMEGA_MATTER    = "+str(self.omegaM)                                  )
            print("            OMEGA_VACUUM    = "+str(self.omega0)                                  )
            print("            HUBBLE PARAM.   = "+str(self.h0)                                      )
            print("            OMEGA_BARYON    = "+str(self.omegaB)                                  )
            print("            SIGMA_8         = "+str(self.sigma8)                                  )
            print("            POWER SPEC.IND. = "+str(self.ns)                                      )
            print("------------------------------------------------------"                           )            
        return


    def redshift(self,snapshot,forceError=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if np.ndim(snapshot) == 0:
            if snapshot not in self.snapshots.index:
                raise ValueError(funcname+"(): Snapshot index outside of allowed range!")
            index = np.argwhere(self.snapshots.index==snapshot)
            redshift = float(self.snapshots.z[index][0])
        else:
            badMask = np.logical_or(snapshot<self.snapshots.index.min(),snapshot>self.snapshots.index.max())
            if any(badMask):
                if forceError:
                    raise ValueError(funcname+"(): Some of specified indices outside of range of snapshots!")
                redshift = np.ones(len(snapshot))*-999.0
                goodMask = np.invert(badMask)
                goodSnapshot = snapshot[goodMask]
                zIndex = np.searchsorted(self.snapshots.index,goodSnapshot)
                np.place(redshift,goodMask,self.snapshots.z[zIndex])
            else:
                zIndex = np.searchsorted(self.snapshots.index,snapshot)
                redshift = self.snapshots.z[zIndex]
        return redshift
                
    def snapshot(self,z,redshift=True,forceError=False):
        if np.ndim(z) == 0:
            snapIndex = np.argmin(np.fabs(self.snapshots.z-z))
            snapshot = int(self.snapshots.index[snapIndex])
        else:
            zz = np.tile(self.snapshots.z,len(z))
            zz = np.matrix(np.reshape(zz,(len(z),-1)))
            zz =  np.fabs(zz - np.transpose(np.matrix(z)))
            snapIndex = np.array(zz.argmin(axis=1))
            snapIndex = snapIndex[:,0]
            snapshot = self.snapshots.index[snapIndex].astype(int)
        if redshift:
            snapshot = self.redshift(snapshot,forceError=forceError)
        return snapshot


    def xmlCosmologicalParametersMethod(self,cosmologicalParametersMethod="simple"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        COS = ET.Element("cosmologyParametersMethod",attrib={"value":cosmologicalParametersMethod})
        ET.SubElement(COS,"HubbleConstant",attrib={"value":str(self.h0*100)})
        ET.SubElement(COS,"OmegaMatter",attrib={"value":str(self.omega0)})
        ET.SubElement(COS,"OmegaDarkEnergy",attrib={"value":str(self.lambda0)})
        ET.SubElement(COS,"OmegaBaryon",attrib={"value":str(self.omegaB)})
        ET.SubElement(COS,"temperatureCMB",attrib={"value":str(self.temperatureCMB)})
        return COS

    def xmlCosmologicalMassVarianceMethod(self,cosmologicalMassVarianceMethod="filteredPower"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        COS = ET.Element("cosmologicalMassVarianceMethod",attrib={"value":cosmologicalMassVarianceMethod})
        ET.SubElement(COS,"sigma_8",attrib={"value":str(self.sigma8)})
        return COS

    def xmlPowerSpectrumPrimordialMethod(self,powerSpectrumPrimordialMethod="powerLaw",\
                                       wavenumberReference=1,running=0):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        PK = ET.Element("powerSpectrumPrimordialMethod",attrib={"value":powerSpectrumPrimordialMethod})
        ET.SubElement(PK,"index",attrib={"value":str(self.ns)})
        ET.SubElement(PK,"wavenumberReference",attrib={"value":str(wavenumberReference)})
        ET.SubElement(PK,"running",attrib={"value":str(running)})
        return PK


        
    def set_parameters(self,cosmologicalParametersMethod="simple",\
                           powerSpectrumPrimordialMethod="powerLaw",wavenumberReference=1,running=0,\
                           mergerTreeFile=None,allTreesExistAtFinalTime=True,\
                           mergerTreeReadOutputTimeSnapTolerance=0.0):
        params = []
        # Set cosmology parameters
        params.append(("cosmologyParametersMethod",cosmologicalParametersMethod))
        params.append(("HubbleConstant",self.h0*100,"cosmologyParametersMethod"))
        params.append(("OmegaMatter",self.omega0,"cosmologyParametersMethod"))
        params.append(("OmegaDarkEnergy",self.lambda0,"cosmologyParametersMethod"))
        params.append(("OmegaBaryon",self.omegaB,"cosmologyParametersMethod"))
        params.append(("temperatureCMB",self.temperatureCMB,"cosmologyParametersMethod"))
        params.append(("sigma_8",self.sigma8,"cosmologicalMassVarianceMethod"))
        params.append(("powerSpectrumPrimordialMethod",powerSpectrumPrimordialMethod))
        params.append(("index",self.ns,"powerSpectrumPrimordialMethod"))
        params.append(("wavenumberReference",wavenumberReference,"powerSpectrumPrimordialMethod"))
        params.append(("running",running,"powerSpectrumPrimordialMethod"))

        # Set merger trees parameters
        if mergerTreeFile is not None:
            params.append(("mergerTreeConstructMethod","read"))
            params.append(("mergerTreeReadFileName",mergerTreeFile))
        if allTreesExistAtFinalTime:
            params.append(("allTreesExistAtFinalTime","true"))
        else:
            params.append(("allTreesExistAtFinalTime","false"))
        params.append(("mergerTreeReadOutputTimeSnapTolerance",mergerTreeReadOutputTimeSnapTolerance))

        return params

        

def locate_simulation_file(simulation):
    funcname = sys._getframe().f_code.co_name
    simfile = None        
    available = "Millennium Millennium2 MS-W7".split()
    # Millennium Simulation
    if fnmatch.fnmatch(simulation.lower(),"millennium") or \
            fnmatch.fnmatch(simulation.lower(),"ms-w1"):
        simfile = pkg_resources.resource_filename(__name__,"data/Simulations/millennium.xml")
    # Millennium 2 Simulation
    if fnmatch.fnmatch(simulation.lower(),"millennium2"):
        simfile = pkg_resources.resource_filename(__name__,"data/Simulations/millennium2.xml")
    # MS-W7 simulation
    if fnmatch.fnmatch(simulation.lower(),"millgas") or \
            fnmatch.fnmatch(simulation.lower(),"ms-w7"):
        simfile = pkg_resources.resource_filename(__name__,"data/Simulations/ms-w7.xml")
    # Planck 256 Mpc/h simulation 
    if fnmatch.fnmatch(simulation.lower(),"p*256*"):
        simfile = pkg_resources.resource_filename(__name__,"data/Simulations/planck256.xml")
    # ERROR if simulation not available
    if simfile is None:
        raise ValueError(funcname+"(): Simulation not recognised! Available simulations are: "+\
                             "\n            "+",".join(available))
    return simfile


