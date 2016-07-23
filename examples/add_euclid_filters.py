#! /usr/bin/env python


import numpy as np
from galacticus.Filters import createFilter


vBandFile = "/Users/aim/Codes/Galacticus/v0.9.4/data/filters/Buser_V.xml"


name = "EUCLID_J"
longname = "JE Euclid NISP J-band."
description = longname +" Euclid minimum requirements for the transmission in the NISP optics, including filter, QE, telescope respone and End Of Life contamination."
origin = "Jerome Amiaux"
dtype = [("wavelength",float),("response",float)]
ifile = "/Users/aim/Projects/SEDS/GALFORM_filters/euclid_filter_JE.dat"
data = np.loadtxt(ifile,dtype=dtype,usecols=range(2))

createFilter(name+".xml",name,data,description=description,origin=origin,url=None,\
                     effectiveWavelength=None,vegaOffset=None,vBandFilter=vBandFile)



name = "EUCLID_H"
longname = "Euclid NISP H-band."
description = longname +" Euclid minimum requirements for the transmission in the NISP optics, including filter, QE, telescope respone and End Of Life contamination."
origin = "Jerome Amiaux"
dtype = [("wavelength",float),("response",float)]
ifile = "/Users/aim/Projects/SEDS/GALFORM_filters/euclid_filter_HE.dat"
data = np.loadtxt(ifile,dtype=dtype,usecols=range(2))

createFilter(name+".xml",name,data,description=description,origin=origin,url=None,\
                     effectiveWavelength=None,vegaOffset=None,vBandFilter=vBandFile)



name = "EUCLID_Y"
longname = "Euclid NISP Y-band."
description = longname +" Euclid minimum requirements for the transmission in the NISP optics, including filter, QE, telescope respone and End Of Life contamination."
origin = "Jerome Amiaux"
dtype = [("wavelength",float),("response",float)]
ifile = "/Users/aim/Projects/SEDS/GALFORM_filters/euclid_filter_YE.dat"
data = np.loadtxt(ifile,dtype=dtype,usecols=range(2))

createFilter(name+".xml",name,data,description=description,origin=origin,url=None,\
                     effectiveWavelength=None,vegaOffset=None,vBandFilter=vBandFile)


name = "EUCLID_VIS"
longname = "Euclid VIS CCD detectors."
description = longname +" Euclid minimum requirements for the quantum efficiency of the VIS CCD detectors, including telescope response and End of Life contamination."
origin = "Jerome Amiaux"
dtype = [("wavelength",float),("response",float)]
ifile = "/Users/aim/Projects/SEDS/GALFORM_filters/euclid_filter_VE.dat"
data = np.loadtxt(ifile,dtype=dtype,usecols=range(2))

createFilter(name+".xml",name,data,description=description,origin=origin,url=None,\
                     effectiveWavelength=None,vegaOffset=None,vBandFilter=vBandFile)







