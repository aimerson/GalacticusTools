#! /usr/bin/env python

import sys,fnmatch
import numpy as np
import scipy as sp
from scipy.constants import c,constants
from scipy.integrate import romberg
from .constants import Pi,massSolar,Parsec,mega

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
        self.H0SI = self.H0*constants.kilo/self.Mpc
        self.HubbleTime = (self.Mpc/(self.H0*constants.kilo))/self.Gyr
        self.HubbleDistance = c/constants.kilo/self.H0
        self.HubbleVolume = self.HubbleDistance**3

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
        result = self.H0*self.E(z)
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
        result = (c/self.H0SI)/np.sqrt(result)/self.Mpc
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

        Note: equations from Mo, van den Bosch & White (2010) Ch.3 Eq. 3.96-3.99
        
        """
        a = 1.0/(1.0+z)
        result = None
        if np.fabs(self.lambda0) < 1.0e-9: # lambda0 = 0
            if self.omega0 == 1.0 : 
                # Einstein de Sitter Universe (Mo et al. 2010, Eq. 3.96)
                result = self.HubbleTime*(2.0/3.0)*(a**(3.0/2.0))
            elif self.omega0 < 1:
                # Open Universe with lambda0 = 0 and omega0 < 1 (Mo et al. 2010, Eq. 3.97)
                factor1 = self.HubbleTime*self.omega0/2.0/((1.0-self.omega0)**(3.0/2.0))
                factor2 = 2.0*np.sqrt((1-self.omega0)*(self.omega0*z+1.0))/self.omega0/(1.0+z)
                factor3 = -np.arccosh((self.omega0*z-self.omega0+2.0)/(self.omega0*(1.0+z)))
                result = factor1*(factor2+factor3)
            else:
                # Closed Universe with lambda = 0 and omega > 1 (Mo et al. 2010, Eq. 3.98)
                factor1 = self.HubbleTime*self.omega0/2.0/((self.omega0-1.0)**(3.0/2.0))                
                factor2 = -2.0*np.sqrt((self.omega0-1.0)*(self.omega0*z+1.0))/self.omega0/(1.0+z)
                factor3 = np.arccos((self.omega0*z-self.omega0+2.0)/(self.omega0*(1.0+z)))
                result = factor1*(factor2+factor3)
        else:
            # Flat Universe with lambda0 + omega0 = 1 (Mo et al. 2010, Eq. 3.99)
            if np.fabs(self.lambda0+self.omega0-1.0) < 1.0e-9:
                factor1 = self.HubbleTime*(2.0/3.0)/np.sqrt(self.lambda0)
                factor2 = np.sqrt(self.lambda0*(a**3)) + np.sqrt(self.lambda0*(a**3)+self.omega0)
                result = factor1*np.log(factor2/np.sqrt(self.omega0))
        return result
            
            
    def lookback_time(self,z=0.0):
        """
        lookback_time(): Returns the lookback time (in Gyr) to 
                         redshift, z.
        
        USAGE: lookback_time(z)
        
        """
        t = self.age_of_universe(0.0) - self.age_of_universe(z)
        return t

    
    def comoving_transverse_distance(self,z=0.0):
        """
        comoving_transverse_distance(): Returns the transverse comoving distance (in Mpc/h or) 
                                        Mpc) corresponding to redshift, z.

        USAGE: comoving_transverse_distance(z)
                                        

        Note: from Hogg (1999) Eq.16.
        """
        if self.omegak > 0:
            result = self.HubbleDistance
            result *= np.sinh(np.sqrt(self.omegak)*self.comoving_distance(z)/self.HubbleDistance)
            result /= np.sqrt(self.omegak)
        elif self.omegak < 0:
            result = self.HubbleDistance
            result *= np.sin(np.sqrt(np.fabs(self.omegak))*self.comoving_distance(z)/self.HubbleDistance)
            result /= np.sqrt(np.fabs(self.omegak))
        else:
            result = self.comoving_distance(z)
        return result


    def angular_diameter_distance(self,z=0.0):
        """
        angular_diameter_distance(): Returns the angular diameter
                                     distance (in Mpc/h) corresponding
                                     to redshift, z.
        
        USAGE: angular_diameter_distance(z)    

        Note: from Hogg (1999) Eq.18

        """
        return self.comoving_transverse_distance(z)/(1.0+z)


    def angular_scale(self,z=0.0):
        """
        angular_scale(): Returns the angular scale (in kpc/arcsec)
                         corresponding to redshift, z.
        
        USAGE: angular_scale(z)
        
        """
        da = self.angular_diameter_distance(z)
        a = da/206.26480
        return a
    

    def angular_distance_separation(self,z1,z2):
        """
        angular_distance_separation(): Returns the angular separation between two 
                                       objects at redshifts z1 and z2.
        
        USAGE: angular_distance_separation(z1,z2)
        
        NOTES: From Hogg (1999) Eq.19
               Assumes omegaK >= 0 (returns None for omegaK < 0)
               Assumes z1 and z2 have equal dimensions
        """
        result = None
        if self.omegak >= 0.0:
            DM1 = self.comoving_transverse_distance(z1)
            DM2 = self.comoving_transverse_distance(z2)
            result = DM2*np.sqrt(1.0*self.omegak*((DM1/self.HubbleDistance)**2))
            result += -DM1*np.sqrt(1.0*self.omegak*((DM2/self.HubbleDistance)**2))
            result *= 1.0/(1.0+z2)
        return result



    def luminosity_distance(self,z=0.0):
        """
        luminosity_distance(): Returns the luminosity distance
                               (in Mpc/h) corresponding to a
                               redshift, z.
        
        USAGE: luminosity_distance(z)
        
        """
        da = self.angular_diameter_distance(z)*self.Mpc/(c/self.H0SI)
        dL = (c/self.H0SI)*da*((1.0+z)**2)/self.Mpc
        return dL
    

    def comoving_volume(self,z=0.0):
        """
        comoving_volume(): Returns the comoving volume (in Mpc^3)
                           contained within a sphere extending out
                           to redshift, z.
        
        USAGE: comoving_volume(z)
        
        Note: From Hogg (1999) Eq.29

        """

        DM = self.comoving_transverse_distance(z)        
        if self.omegak > 0.0:
            DMDH = DM/self.HubbleDistance            
            factor1 = 4.0*Pi*self.HubbleVolume/2.0/self.omegak
            factor2 = DMDH*np.sqrt(1.0+self.omegak*(DMDH**2))
            factor3 = -np.arcsinh(np.sqrt(np.fabs(self.omegak))*DMDH)/np.sqrt(np.fabs(self.omegak))
            result = factor1*(factor2+factor3)
        elif self.omegak < 0.0:
            DMDH = DM/self.HubbleDistance            
            factor1 = 4.0*Pi*self.HubbleVolume/2.0/self.omegak
            factor2 = DMDH*np.sqrt(1.0+self.omegak*(DMDH**2))
            factor3 = -np.arcsin(np.sqrt(np.fabs(self.omegak))*DMDH)/np.sqrt(np.fabs(self.omegak))
            result = factor1*(factor2+factor3)
        else:
            result = 4.0*Pi*(DM**3)/3.0
        return result


    def dVdz(self,z=0.0):
        """
        dVdz() : Returns the comoving volume element dV/dz
                 at redshift, z, for all sky.
        
        dV = (c/H0)*(1+z)**2*D_A**2/E(z) dz dOmega
        
        USAGE: dVdz(z)

        """
        dA = self.angular_diameter_distance(z)
        dV = self.HubbleDistance*(dA**2)*((1.0+z)**2)/self.E(z)
        return dV*4.0*Pi
    

    def band_corrected_distance_modulus(self,z=0.0):
        """
        band_corrected_distance_modulus(): returns the Band Corrected
                              Distance Modulus (BCDM) at redshift, z.
        
        USAGE: band_corrected_distance_modulus(z)

        NOTE from Galacticus manual:
        The luminosity computed in this way is that in the galaxy rest
        frame using a filter blueshifted to the galaxy's redshift. This means
        that to compute an apparent magnitude you must add not only the
        distance modulus, but a factor of 2.5 log10(1 + z) to account for
        compression of photon frequencies.
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

    #
    # Functions for N-body simulations
    #
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
    
    def __init__(self,year,h_independent=True,radiation=False,zmax=20.0,nzmax=10000):
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
                                       zmax=zmax,nzmax=nzmax,h_independent=h_independent)
        return



def adjustHubble(values,hIn,hOut,datatype,verbose=False):
    funcname = sys._getframe().f_code.co_name    
    # Get type of data to convert
    if fnmatch.fnmatch(datatype.lower(),"mag*"):
        dtype = "magnitude"
        result = values - 5.0*np.log10(hOut/hIn)
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
        


def wavelengthToRedshift(obsv,emit):
    return (obsv/emit) - 1.0

def redshiftToWavelength(z,emit):
    return (1.0+z)*emit
    
def MpcToCM(r):
    return r*Parsec*mega*100.0





    
        


    
