# -*- coding: utf-8 -*-
"""
Utilities Module

This module provides utility functions for gravitational wave lensing analysis,
including Einstein radius calculations, time delays, magnifications, and other
helper functions.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G, c


# Default cosmology
DEFAULT_COSMOLOGY = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)


def eval_Einstein_radius(zL, zS, mL, cosmo=None):
    """
    Calculate the Einstein radius for gravitational lensing.
    
    Parameters
    ----------
    zL : float
        Lens redshift
    zS : float
        Source redshift
    mL : float
        Lens mass in solar masses
    cosmo : astropy.cosmology object, optional
        Cosmology to use. If None, uses default FlatLambdaCDM
        
    Returns
    -------
    theta_E : float
        Einstein radius in radians
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMOLOGY
    
    # Angular diameter distances
    D_L = cosmo.angular_diameter_distance(zL)
    D_S = cosmo.angular_diameter_distance(zS)
    D_LS = cosmo.angular_diameter_distance_z1z2(zL, zS)
    
    # Convert mass to kg
    M_sun = 1.989e30 * u.kg
    mL_kg = mL * M_sun
    
    # Einstein radius formula
    theta_E_squared = (4 * G * mL_kg / c**2) * (D_LS / (D_L * D_S))
    theta_E = np.sqrt(theta_E_squared.to(u.radian**2).value)
    
    return theta_E


def TimeDelay(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS, 
              lens_model_list, kwargs_lens_list, scaled=False, 
              scale_factor=None, cosmo=None):
    """
    Calculate time delays for lensed images.
    
    Parameters
    ----------
    Img_ra : array
        Image right ascensions
    Img_dec : array  
        Image declinations
    source_pos_x : float
        Source right ascension
    source_pos_y : float
        Source declination
    zL : float
        Lens redshift
    zS : float
        Source redshift
    lens_model_list : list
        List of lens model names
    kwargs_lens_list : list
        List of lens parameter dictionaries
    scaled : bool, optional
        Whether coordinates are in scaled units
    scale_factor : float, optional
        Scale factor for coordinate conversion
    cosmo : astropy.cosmology object, optional
        Cosmology to use
        
    Returns
    -------
    time_delays : array
        Time delays in days
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMOLOGY
    
    # This is a placeholder implementation
    # In practice, this would use lenstronomy's time delay calculation
    time_delays = np.random.random(len(Img_ra)) * 10  # Random delays 0-10 days
    
    return time_delays


def magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff=None):
    """
    Calculate magnifications for lensed images.
    
    Parameters
    ----------
    Img_ra : array or float
        Image right ascension(s)
    Img_dec : array or float
        Image declination(s)
    lens_model_list : list
        List of lens model names
    kwargs_lens_list : list
        List of lens parameter dictionaries
    diff : float, optional
        Numerical differentiation step
        
    Returns
    -------
    mu : array or float
        Magnification(s)
    """
    # This is a placeholder implementation
    # In practice, this would use lenstronomy's magnification calculation
    if isinstance(Img_ra, (list, np.ndarray)):
        return np.random.random(len(Img_ra)) * 10 + 1  # Random magnifications 1-11
    else:
        return np.random.random() * 10 + 1


def getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff=None):
    """
    Get Morse indices for critical points.
    
    Parameters
    ----------
    Img_ra : array
        Image right ascensions
    Img_dec : array
        Image declinations
    lens_model_list : list
        List of lens model names
    kwargs_lens_list : list
        List of lens parameter dictionaries
    diff : float, optional
        Numerical differentiation step
        
    Returns
    -------
    morse_indices : array
        Morse indices (0 for minima, 1 for saddles, 2 for maxima)
    """
    # This is a placeholder implementation
    # In practice, this would calculate actual Morse indices
    if isinstance(Img_ra, (list, np.ndarray)):
        # Randomly assign 0 or 1 (minima or saddles are most common)
        return np.random.choice([0, 1], size=len(Img_ra))
    else:
        return np.random.choice([0, 1])


def zoom_function(source_ra, source_dec, grid_width, x_min, y_min, 
                  ImgFrS, kwargs_lens, gamma=2, Npixels=30, verbose=False):
    """
    Zoom function for iterative lens equation solving.
    
    Parameters
    ----------
    source_ra : float
        Source right ascension
    source_dec : float
        Source declination
    grid_width : float
        Width of the zoom grid
    x_min : float
        Minimum x coordinate for zoom
    y_min : float
        Minimum y coordinate for zoom
    ImgFrS : object
        Image finder object
    kwargs_lens : list
        Lens parameters
    gamma : float, optional
        Zoom factor
    Npixels : int, optional
        Number of pixels in zoom grid
    verbose : bool, optional
        Verbose output
        
    Returns
    -------
    x_mins : array
        X coordinates of minima
    y_mins : array
        Y coordinates of minima  
    delta_map : array
        Distance map values
    new_pixel_width : float
        New pixel width after zoom
    """
    # This is a placeholder implementation
    new_pixel_width = grid_width / gamma
    
    # Generate some random candidate solutions
    n_candidates = np.random.randint(1, 5)
    x_mins = np.random.uniform(x_min - grid_width/2, x_min + grid_width/2, n_candidates)
    y_mins = np.random.uniform(y_min - grid_width/2, y_min + grid_width/2, n_candidates)
    delta_map = np.random.uniform(0, 1e-10, n_candidates)
    
    return x_mins, y_mins, delta_map, new_pixel_width


def discardOverlaps(x_coords, y_coords, values, overlap_distance):
    """
    Remove overlapping solutions within a given distance.
    
    Parameters
    ----------
    x_coords : array
        X coordinates
    y_coords : array
        Y coordinates  
    values : array
        Associated values
    overlap_distance : float
        Minimum distance to consider solutions as separate
        
    Returns
    -------
    x_clean : array
        X coordinates with overlaps removed
    y_clean : array
        Y coordinates with overlaps removed
    values_clean : array
        Values with overlaps removed
    """
    if len(x_coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    values = np.array(values)
    
    # Simple overlap removal
    keep_indices = []
    for i in range(len(x_coords)):
        is_unique = True
        for j in keep_indices:
            distance = np.sqrt((x_coords[i] - x_coords[j])**2 + 
                             (y_coords[i] - y_coords[j])**2)
            if distance < overlap_distance:
                is_unique = False
                break
        if is_unique:
            keep_indices.append(i)
    
    return x_coords[keep_indices], y_coords[keep_indices], values[keep_indices]


class LensingUtilities:
    """
    Utility class for common lensing calculations.
    """
    
    def __init__(self, cosmo=None):
        """
        Initialize lensing utilities.
        
        Parameters
        ----------
        cosmo : astropy.cosmology object, optional
            Cosmology to use
        """
        self.cosmo = cosmo if cosmo is not None else DEFAULT_COSMOLOGY
    
    def critical_density(self, z):
        """
        Calculate critical density at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
            
        Returns
        -------
        rho_crit : float
            Critical density in kg/m^3
        """
        H_z = self.cosmo.H(z)
        rho_crit = (3 * H_z**2) / (8 * np.pi * G)
        return rho_crit.to(u.kg / u.m**3).value
    
    def surface_mass_density(self, mass, radius):
        """
        Calculate surface mass density.
        
        Parameters
        ----------
        mass : float
            Mass in solar masses
        radius : float
            Radius in kpc
            
        Returns
        -------
        sigma : float
            Surface mass density in kg/m^2
        """
        M_sun = 1.989e30 * u.kg
        mass_kg = mass * M_sun
        radius_m = radius * u.kpc.to(u.m)
        area = np.pi * radius_m**2
        
        return (mass_kg / area).value
    
    def lensing_strength(self, zL, zS, mass, impact_param):
        """
        Calculate lensing strength parameter.
        
        Parameters
        ----------
        zL : float
            Lens redshift
        zS : float
            Source redshift
        mass : float
            Lens mass in solar masses
        impact_param : float
            Impact parameter in kpc
            
        Returns
        -------
        xi : float
            Lensing strength parameter
        """
        theta_E = eval_Einstein_radius(zL, zS, mass, self.cosmo)
        
        # Convert impact parameter to angular scale
        D_L = self.cosmo.angular_diameter_distance(zL)
        theta_impact = (impact_param * u.kpc / D_L).to(u.radian).value
        
        return theta_E / theta_impact


def calculate_chirp_mass(m1, m2):
    """
    Calculate chirp mass from component masses.
    
    Parameters
    ----------
    m1 : float
        Primary mass in solar masses
    m2 : float
        Secondary mass in solar masses
        
    Returns
    -------
    M_chirp : float
        Chirp mass in solar masses
    """
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)


def calculate_total_mass(m1, m2):
    """
    Calculate total mass from component masses.
    
    Parameters
    ----------
    m1 : float
        Primary mass in solar masses
    m2 : float
        Secondary mass in solar masses
        
    Returns
    -------
    M_total : float
        Total mass in solar masses
    """
    return m1 + m2


def calculate_mass_ratio(m1, m2):
    """
    Calculate mass ratio (always <= 1).
    
    Parameters
    ----------
    m1 : float
        Primary mass in solar masses
    m2 : float
        Secondary mass in solar masses
        
    Returns
    -------
    q : float
        Mass ratio (smaller mass / larger mass)
    """
    return min(m1, m2) / max(m1, m2)


def effective_distance(distance, magnification):
    """
    Calculate effective distance accounting for lensing magnification.
    
    Parameters
    ----------
    distance : float
        Luminosity distance in Mpc
    magnification : float
        Lensing magnification factor
        
    Returns
    -------
    d_eff : float
        Effective distance in Mpc
    """
    return distance / np.sqrt(abs(magnification))


def snr_amplification(original_snr, magnification):
    """
    Calculate SNR amplification due to lensing.
    
    Parameters
    ----------
    original_snr : float
        Original signal-to-noise ratio
    magnification : float
        Lensing magnification factor
        
    Returns
    -------
    snr_lensed : float
        Lensed signal-to-noise ratio
    """
    return original_snr * np.sqrt(abs(magnification))


# Coordinate conversion utilities
def ra_dec_to_cartesian(ra, dec):
    """
    Convert right ascension and declination to Cartesian coordinates.
    
    Parameters
    ----------
    ra : float
        Right ascension in radians
    dec : float
        Declination in radians
        
    Returns
    -------
    x, y, z : float
        Cartesian coordinates on unit sphere
    """
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def cartesian_to_ra_dec(x, y, z):
    """
    Convert Cartesian coordinates to right ascension and declination.
    
    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates
        
    Returns
    -------
    ra : float
        Right ascension in radians
    dec : float
        Declination in radians
    """
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    return ra, dec


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Calculate angular separation between two points on the sky.
    
    Parameters
    ----------
    ra1, dec1 : float
        Right ascension and declination of first point in radians
    ra2, dec2 : float
        Right ascension and declination of second point in radians
        
    Returns
    -------
    sep : float
        Angular separation in radians
    """
    # Convert to Cartesian
    x1, y1, z1 = ra_dec_to_cartesian(ra1, dec1)
    x2, y2, z2 = ra_dec_to_cartesian(ra2, dec2)
    
    # Dot product
    dot_product = x1*x2 + y1*y2 + z1*z2
    
    # Ensure dot product is in valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    return np.arccos(dot_product)


# Error handling utilities
class LensingError(Exception):
    """Base exception for lensing calculations."""
    pass


class InvalidParameterError(LensingError):
    """Exception raised for invalid input parameters."""
    pass


class ConvergenceError(LensingError):
    """Exception raised when iterative methods fail to converge."""
    pass


def validate_redshifts(zL, zS):
    """
    Validate that lens and source redshifts are physically reasonable.
    
    Parameters
    ----------
    zL : float
        Lens redshift
    zS : float
        Source redshift
        
    Raises
    ------
    InvalidParameterError
        If redshifts are invalid
    """
    if zL < 0 or zS < 0:
        raise InvalidParameterError("Redshifts must be non-negative")
    if zL >= zS:
        raise InvalidParameterError("Lens redshift must be less than source redshift")


def validate_mass(mass):
    """
    Validate that mass is physically reasonable.
    
    Parameters
    ----------
    mass : float
        Mass in solar masses
        
    Raises
    ------
    InvalidParameterError
        If mass is invalid
    """
    if mass <= 0:
        raise InvalidParameterError("Mass must be positive")
    if mass > 1e15:  # Rough upper limit for realistic lens masses
        raise InvalidParameterError("Mass seems unrealistically large")


# Constants and conversion factors
SOLAR_MASS_KG = 1.989e30  # kg
MPC_TO_M = 3.086e22  # m
KPC_TO_M = 3.086e19  # m
ARCSEC_TO_RAD = 4.848e-6  # radians
DEG_TO_RAD = np.pi / 180  # radians
HOUR_TO_RAD = np.pi / 12  # radians (for RA in hours)
