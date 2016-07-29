#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import scipy as sp
from scipy.constants import c,constants
from scipy.integrate import romberg
from .constants import Pi,massSolar,Parsec

class Cosmology(object):
    """
    Cosmology: class to compute distances and times in 
               a Universe with a given cosmology.
    
    List of functions:

    report_parameters() : report back parameters for specified cosmology
    comoving_distance() : calculates the comoving distance at
                          redshift, z
    redshift_at_distance() : calculates the redshift at comoving
                             disance, r
    age_of_universe() : calculates the age of the Universe at
                        redshift, z
    lookback_time() : calculates lookback time to given redshift,
                      z
    angular_diamater_distance() : calculates the angular diameter
                                  distance a redshift, z
    angular_scale() : calculates the angular scale at redshift, z
    luminosity_distance() : calculates the luminosity distance at
                            redshift, z
    comving_volume() : calculates the comoving volume contained
                       within a sphere extending out to redshift,
                       z
    dVdz() :  calculates dV/dz at redshift, z
    H() : return Hubble constant as measured at redshift, z
    E() : returns Peebles' E(z) function at redshift, z, for
          specified cosmology

    NOTE: this module requires the numpy and scipy libraries.

    Based upon the 'Cosmology Calculator' (Wright, 2006, PASP,
    118, 1711) and Fortran 90 code written by John Helly.
    
    """
    
    def __init__(self,omega0=0.25,lambda0=0.75,omegab=0.045,h0=0.73,sigma8=0.9,ns=1.0,\
                     radiation=False,zmax=20.0,nzmax=10000,h_independent=True):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name

        # Store cosmological parameters
        self.omega0 = omega0
        self.lambda0 = lambda0
        self.omegab = omegab
        self.h0 = h0
        self.sigma8 = sigma8
        self.ns = ns
        if radiation:            
            self.omegar = (4.165e-5)/(self.h0**2)
        else:
            self.omegar = 0.0            
        self.omegak = 1.0 - (self.omega0 + self.lambda0 + self.omegar)

        # Store value for Hubble Constant
        if h_independent:
            self.H0 = 100
        else:
            self.H0 = 100.0*self.h0
        self.h_independent = h_independent
        
        # Define useful constants/conversions
        self.Mpc = constants.mega*Parsec        
        self.Gyr = constants.giga*constants.year
        self._kmpersec_to_mpchpergyr = constants.kilo*(self.Gyr/self.Mpc)*self.h0                
        self.H100 = 100.0*constants.kilo/self.Mpc
        self.invH0 = (self.Mpc/(100.0*constants.kilo))/self.Gyr
        self.HubbleDistance = c/self.H100

        # Compute critical density
        self.criticalDensity = (3.0*(100**2)/8.0/Pi/constants.G)
        self.criticalDensity *=(constants.kilo/self.Mpc)**2
        self.criticalDensity /= massSolar/(self.Mpc**3)

        # Set up array of redshift vs. comoving distance for
        # interpolation for other properties
        self._nzmax = nzmax
        self._zmax = zmax
        self._r_comoving = np.zeros(self._nzmax)
        self._dz = self._zmax/float(self._nzmax)
        self._redshift = np.arange(0.0,self._zmax,self._dz)
        self._inv_dz = 1.0/self._dz
        self._initialize_redshift_array = True

        return


    def report_parameters(self):
        report = "\nCOSMOLOGY:\n" + \
            "   Omega_M = {0:5.3f}\n".format(self.omega0) + \
            "   Omega_b = {0:5.3f}\n".format(self.omegab) + \
            "   Omega_V = {0:5.3f}\n".format(self.lambda0) + \
            "   h       = {0:5.3f}\n".format(self.h0) + \
            "   sigma_8 = {0:5.3f}\n".format(self.sigma8) + \
            "   n_s     = {0:5.3f}\n".format(self.ns) + \
            "   Omega_R = {0:5.3e}\n".format(self.omegar) + \
            "   Omega_k = {0:5.3f}\n".format(self.omegak)
        report = "-"*30 + report + "-"*30 + "\n"
        print(report)
        return
    

    def E(self,z=0.0):
        """
        E(z): Peebles' E(z) function.
        
        """
        a = 1.0/(1.0+z)
        result = self.omegak*(a**-2) + self.lambda0 + \
                 self.omega0*(a**-3) + self.omegar*(a**-4)
        return np.sqrt(result)


    def H(self,z=0.0):
        """
        H(z): Function to return the Hubble parameter as measured
              by an observer at redshift, z.
        """
        result = 100.0*self.E(z)
        return result

    
    def f(self,z=0.0):
        """
        f(z): Function relating comoving distance to redshift.
              Integrating f(z)dz from 0 to z' gives comoving
              distance r(z'). Result is in Mpc/h.
        
        Note: uses global cosmology variables.          
        """
        a = 1.0/(1.0+z)
        result = self.omegak*(a**-2) + self.lambda0 + \
                 self.omega0*(a**-3) + self.omegar*(a**-4)
        result = (c/self.H100)/np.sqrt(result)/self.Mpc
        return result


    def _init_redshift_array(self):
        for i in range(1,len(self._redshift)):
            z1 = self._redshift[i-1]
            z2 = self._redshift[i]
            self._r_comoving[i] = self._r_comoving[i-1] + \
                                  romberg(self.f,z1,z2)
        self._initialize_redshift_array = False
        return

    
    def comoving_distance(self,z=0.0):
        """
        comoving_distance(): Returns the comoving distance (in Mpc/h)
                             corresponding to redshift, z.
        
        USAGE: comoving_distance(z)
        
        """
        if self._initialize_redshift_array:
            self._init_redshift_array()
        return np.interp(z,self._redshift,self._r_comoving)

    
    def redshift_at_distance(self,r=0.0):
        """
        redshift_at_distance(): Returns the redshift corresponding
                                to comoving distance, r (in Mpc/h).
            
        USAGE: redshift_at_distance(z)
        
        """
        if self._initialize_redshift_array:
            self._init_redshift_array()
        return np.interp(r,self._r_comoving,self._redshift)
    
    
    def age_of_universe(self,z=0.0):
        """
        age_of_universe(): Returns the age of the Universe (in Gyr) at
                           a redshift, z, for the given cosmology.
        
        USAGE: age_of_universe(z)
        
        """
        a = 1.0/(1.0+z)
        if(self.omega0 >= 0.99999): # Einstein de Sitter Universe
            result = self.invH0*2.0*np.sqrt(a)/(3.0*self.h0)
        else:
            if(self.lambda0 <= 0.0): # Open Universe
                zplus1 = 1.0/a
                result1 = self.omega0/(2.0*self.h0*(1-self.omega0)**1.5)
                result2 = 2.0*np.sqrt(1.0-self.omega0)*np.sqrt(self.omega0*(zplus1-1.0)+1.0)
                result3 = np.arccosh((self.omega0*(zplus1-1.0)-self.omega0+2.0)/(self.omega0*zplus1))
                result = self.invH0*result1*(result2/result3)
            else: # Flat Universe with non-zero Cosmological Constant
                result1 = (2.0/(3.0*self.h0*np.sqrt(1.0-self.omega0)))
                result2 = np.arcsinh(np.sqrt((1.0/self.omega0-1.0)*a)*a)
                result = self.invH0*result1*result2
        return result
            
            
    def lookback_time(self,z=0.0):
        """
        lookback_time(): Returns the lookback time (in Gyr) to 
                         redshift, z.
        
        USAGE: lookback_time(z)
        
        """
        t = self.age_of_universe(0.0) - self.age_of_universe(z)
        return t


    def angular_diameter_distance(self,z=0.0):
        """
        angular_diameter_distance(): Returns the angular diameter
                                     distance (in Mpc/h) corresponding
                                     to redshift, z.
        
        USAGE: angular_diameter_distance(z)    

        """
        dr = self.comoving_distance(z)*self.Mpc/(c/self.H100)
        x = np.sqrt(np.abs(self.omegak))*dr
        if np.ndim(x) > 0:
            ratio = np.ones_like(x)*-1.00
            mask = (x > 0.1)
            y = x[np.where(mask)]
            if(self.omegak > 0.0):
                np.place(ratio,mask,0.5*(np.exp(y)-np.exp(-y))/y)
            else:
                np.place(ratio,mask,np.sin(y)/y)
            mask = (x <= 0.1)
            y = x[np.where(mask)]**2
            if(self.omegak < 0.0): 
                y = -y
            np.place(ratio,mask,1.0 + y/6.0 + (y**2)/120.0)
        else:        
            ratio = -1.0
            if(x > 0.1):
                if(self.omegak > 0.0):
                    ratio = 0.5*(np.exp(x)-np.exp(-x))/x
                else:
                    ratio = np.sin(x)/x
            else:
                y = x**2
                if(self.omegak < 0.0): 
                    y = -y
                ratio = 1.0 + y/6.0 + (y**2)/120.0
        dt = ratio*dr/(1.0+z)
        dA = (c/self.H100)*dt/self.Mpc
        return dA


    def angular_scale(self,z=0.0):
        """
        angular_scale(): Returns the angular scale (in kpc/arcsec)
                         corresponding to redshift, z.
        
        USAGE: angular_scale(z)
        
        """
        da = self.angular_diameter_distance(z)
        a = da/206.26480
        return a
    

    def luminosity_distance(self,z=0.0):
        """
        luminosity_distance(): Returns the luminosity distance
                               (in Mpc/h) corresponding to a
                               redshift, z.
        
        USAGE: luminosity_distance(z)
        
        """
        da = self.angular_diameter_distance(z)*self.Mpc/(c/self.H100)
        dL = (c/self.H100)*da*((1.0+z)**2)/self.Mpc
        return dL
    

    def comoving_volume(self,z=0.0):
        """
        comoving_volume(): Returns the comoving volume (in Mpc^3)
                           contained within a sphere extending out
                           to redshift, z.
        
        USAGE: comoving_volume(z)
        
        """
        dr = self.comoving_distance(z)*self.Mpc/(c/self.H100)
        x = np.sqrt(np.abs(self.omegak))*dr
        if np.ndim(z) > 0:
            ratio = np.ones_like(z)*-1.0
            mask = (x > 0.1)
            y = x[np.where(mask)]
            if(self.omegak > 0.0):
                rat = (0.125*(np.exp(2.0*y)-np.exp(-2.0*y))-y/2.0)
            else:
                rat = (y/2.0 - np.sin(2.0*y)/4.0)
            np.place(ratio,mask,rat/((y**3)/3.0))
            mask = (x <= 0.1)
            y = x[np.where(mask)]**2
            if(self.omegak < 0.0): 
                y = -y
            np.place(ratio,mask,1.0 + y/5.0 + (y**2)*(2.0/105.0))
        else:  
            ratio = -1.0
            if(x > 0.1):
                if(self.omegak > 0.0):
                    ratio = (0.125*(np.exp(2.0*x)-np.exp(-2.0*x))-x/2.0)
                else:
                    ratio = (x/2.0 - np.sin(2.0*x)/4.0)
                ratio = ratio/((x**3)/3.0)
            else:
                y = x**2
                if(self.omegak < 0.0): 
                    y = -y
                ratio = 1.0 + y/5.0 + (y**2)*(2.0/105.0)
        vol = 4.0*np.pi*ratio*(((c/self.H100)*dr/self.Mpc)**3)/3.0
        return vol


    def dVdz(self,z=0.0):
        """
        dVdz() : Returns the comoving volume element dV/dz
                 at redshift, z, for all sky.
        
        dV = (c/H100)*(1+z)**2*D_A**2/E(z) dz dOmega
        
        f(z) = (c/H100)/E(z)
        
        ==> dV/dz(z,all sky) = 4*PI*f(z)*(1+z)**2*D_A**2
             
        
        USAGE: dVdz(z)

        """
        dA = self.angular_diameter_distance(z)
        return self.f(z)*(dA**2)*((1.0+z)**2)*4.0*np.pi
    

    def band_corrected_distance_modulus(self,z=0.0):
        """
        band_corrected_distance_modulus(): returns the Band Corrected
                              Distance Modulus (BCDM) at redshift, z.
        
        USAGE: band_corrected_distance_modulus(z)

        NOTE from Galacticus manual:
        The luminosity computed in this way is that in the galaxy rest
        frame using a filter blueshifted to the galaxy’s redshift. This means
        that to compute an apparent magnitude you must add not only the
        distance modulus, but a factor of 2.5 log10(1 + z) to account for
        compression of photon frequencies.

        There is no h dependence as we work always in length units of Mpc/h 
        such that our absolute magnitudes are really Mabs-5logh and no 
        additional h dependence is needed here to get apparent magnitudes 
        that are h independent.
        
        """
        dref = 10.0/constants.mega # 10pc in Mpc
        dL = self.luminosity_distance(z)
        bcdm = 5.0*np.log10(dL/dref) - 2.5*np.log10(1.0+z)
        return bcdm


    def realspace(self,ra,dec,z):
        ra = np.radians(ra)
        dec = np.radians(dec)
        r = self.comoving_distance(z)
        XX = r*np.cos(dec)*np.cos(ra)
        YY = r*np.cos(dec)*np.sin(ra)
        ZZ = r*np.sin(dec)
        return XX,YY,ZZ



    def particleMass(self,boxSize,particlesPerSide):
        numberDensity = (float(particlesPerSide)/float(boxSize))**3
        return self.criticalDensity*self.omega0/numberDensity

    def boxSize(self,particleMass,particlesPerSide):
        boxSize = particleMass*(particlesPerSide**3)
        boxSize /= self.criticalDensity*self.omega0
        return boxSize**(1.0/3.0)
    
    def particlesPerSide(self,boxSize,particleMass):
        return (self.criticalDensity*self.omega0*(boxSize**3)/particleMass)**(1.0/3.0)


class WMAP(Cosmology):
    
    def __init__(self,year,radiation=False,zmax=20.0,nzmax=10000):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.year = year        
        if self.year == 1:
            omega0 = 0.25
            lambda0 = 0.75
            omegab = 0.045
            h0 = 0.73
            sigma8 = 0.9
            ns = 1.0
        elif year == 3:
            pass
        elif year == 5:
            pass
        elif year == 7:
            omega0 = 0.272
            lambda0 = 0.728
            omegab = 0.045
            h0 = 0.702
            sigma8 = 0.807
            ns = 0.961
        elif year == 9:
            pass
        else:
            print("*** ERROR! "+classname+"(): year not recognised!")
            print("           Select one of the following years: 1,3,5,7,9.")                        
        super(WMAP, self).__init__(omega0=omega0,lambda0=lambda0,omegab=omegab,h0=h0,\
                                       sigma8=sigma8,ns=ns,radiation=radiation,\
                                       zmax=zmax,nzmax=nzmax)
        return



def adjustHubble(values,hIn,hOut,datatype,verbose=False):
    funcname = sys._getframe().f_code.co_name    
    # Get type of data to convert
    if fnmatch.fnmatch(datatype.lower(),"mag*"):
        dtype = "magnitude"
        result = values - 5.0*np.log10(hOut,hIn)
    elif fnmatch.fnmatch(datatype.lower(),"lum*"):
        dtype = "luminosity"
        result = values * ((hOut/hIn)**2)
    elif fnmatch.fnmatch(datatype.lower(),"dis*"):
        dtype = "distance"
        result = values * (hIn/hOut)
    elif fnmatch.fnmatch(datatype.lower(),"vol*"):
        dtype = "volume"
        result = values * ((hIn/hOut)**3)
    elif fnmatch.fnmatch(datatype.lower(),"mass*"):
        dtype = "mass"
        result = values * (hIn/hOut)
    elif fnmatch.fnmatch(datatype.lower(),"den*"):
        dtype = "density"
        result = values * ((hOut/hIn)**3)
    else:
        availableTypes = ["magnitude","luminosity","distance","volume","mass","density"]
        report = funcname+"(): Specified type not recognised!\n"
        report = report = "      Available datatypes are: "+", ".join(availableTypes)
        raise ValueError(report)
    if verbose:
        print(funcname+"(): Converted "+dtype+" from h="+str(hIn)+" to h="+str(hOut))    
    return result
        









    
        


    
