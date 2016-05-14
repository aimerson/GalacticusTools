#! /usr/bin/env python

import numpy as np
from .io import GalacticusHDF5
from .Inclination import Get_Inclination
from .constants import Pi
from .constants import megaParsec,massHydrogen,massSolar,hydrogenByMassPrimordial

def Get_Column_Density(galHDF5Obj,z,components=None,spheroidCutoff=0.1,diskHeightRatio=0.1):
    """
    Get_Column_Density(): Calculate column densities for galaxies at specified redshift, considering
                          specified components of galaxies. 

    USAGE:  columnDensities = Get_Column_Density(galHDF5Obj,z,[components][spheroidCutoff],[diskHeightRatio])


            INPUTS:

                  galHDF5Obj      : Instance of GalacticusHDF5 object class.
                  z               : Reshift of interest.
                  components      : List of components to consider. If not specified (None), then will consider
                                    all components, i.e. ['spheroid','disk','total']. (Default value = None).
                  spheroidCutoff  : Cutoff for spheroid radius adopted in calculation for spheroids.
                                    (Default value = 0.1).
                  diskHeightRatio : Ratio of disk height to (?) adopted in calculation for disks.
                                    (Default value = 0.1).

            OUTPUTS:

                  columnDensities : Dictionary of column densities, each stored in Numpy arrays. Keys will
                                    include 'spheroid', 'disk' and 'total'.


    """
    # Set default to calculate column density for all
    # possible components
    if components is None:
        components = ["disk","spheroid","total"]
        
    # Get output for this redshift
    out = galHDF5Obj.selectOutput(z)

    # Spheroid component
    if "spheroid" in map(str.lower,components) or "total" in map(str.lower,components):        
        massGas = np.array(out["nodeData/spheroidMassGas"])
        radius = np.array(out["nodeData/spheroidRadius"])
        sigmaSpheroid = np.copy(np.zeros_like(radius))
        mask = radius > 0.0
        massGas = massGas[mask]
        radius = radius[mask]        
        densityCentral = massGas/(2.0*Pi*radius**3) 
        # Using 0.1*(scale length) for inner spheroid cutoff
        radiusMinimum = spheroidCutoff*radius;
        radiusMinimumDimensionless = radiusMinimum/radius;        
        sigma = 0.5*np.copy(densityCentral)*np.copy(radius)
        sigma *= (-(3.0+2.0*radiusMinimumDimensionless)/\
            (1.0+radiusMinimumDimensionless)**2+\
            2.0*np.log(1.0+1.0/radiusMinimumDimensionless))
        np.place(sigmaSpheroid,mask,np.copy(sigma))
        del massGas,radius,densityCentral
        del radiusMinimum,radiusMinimumDimensionless,sigma
    else:
        sigmaSpheroid = None

    # Disk component
    if "disk" in map(str.lower,components) or "total" in map(str.lower,components):        
        from scipy.special import psi
        massGas = np.array(out["nodeData/diskMassGas"])
        radius = np.array(out["nodeData/diskRadius"])
        sigmaDisk = np.copy(np.zeros_like(radius))
        mask = radius > 0.0
        massGas = massGas[mask]
        radius = radius[mask]        
        densityCentral = massGas/(4.0*Pi*radius**3*diskHeightRatio)
        # Compute column density to center of disk.
        if "inclination" not in galHDF5Obj.availableDatasets(z):
            inclination = Get_Inclination(galHDF5Obj,z)
        else:
            inclination = np.array(out["nodeData/inclination"])*(Pi/180.0)
        inclination = np.fabs(np.tan(inclination))
        inclinationHeight = inclination*diskHeightRatio
        digamma1 = psi(-inclinationHeight/4.0)
        digamma2 = psi(0.5-inclinationHeight/4.0)
        sigma = 0.5*densityCentral*radius*np.sqrt(1.0+1.0/inclination**2)
        sigma *= (inclinationHeight*(digamma1-digamm2)-2.0)
        np.place(sigmaDisk,mask,np.copy(sigma))
        del massGas,radius,densityCentral,inclination,inclinationHeight
        del digamma1,digamma2,sigma
    else:
        sigmaDisk = None
        
    # Evaluate conversion factor from mass column density to hydrogen column density.    
    hecto = 1.00000000000e+02
    hydrogenFactor = hydrogenByMassPrimordial*massSolar/(massHydrogen*(megaParsec*hecto)**2)
    
    # Compute and return the column densities. 
    columnDensities = {}
    if "total" in map(str.lower,components):        
        galHDF5Obj.addDataset(out.name+"/nodeData/","columnDensityDisk",\
                                  np.copy((sigmaSpheroid+sigmaDisk)*hydrogenFactor))        
        columnDensities["total"] = (sigmaSpheroid+sigmaDisk)*hydrogenFactor
    if "spheroid" in map(str.lower,components):        
        galHDF5Obj.addDataset(out.name+"/nodeData/","columnDensitySpheroid",sigmaSpheroid*hydrogenFactor)
        columnDensities["spheroid"] = np.copy(sigmaSpheroid*hydrogenFactor)
        del sigmaSpheroid
    if "disk" in map(str.lower,components):        
        galHDF5Obj.addDataset(out.name+"/nodeData/","columnDensityDisk",sigmaDisk*hydrogenFactor)        
        columnDensities["disk"] = np.copy(sigmaDisk*hydrogenFactor)
        del sigmaDisk

    return columnDensities
