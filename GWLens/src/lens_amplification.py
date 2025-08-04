# -*- coding: utf-8 -*-
"""
Amplification Module

This module provides functions for computing geometrical optics amplification
factors for gravitational wave lensing.
"""

import numpy as np
from .utils import TimeDelay, getMinMaxSaddle, magnifications


def amplification_from_data(frequencies, mu, td, n):
    """
    Computes the geometrical optics amplification F(f) over a frequency band,
    given a set of magnifications, time delays and Morse indices

    Parameters
    ----------
    frequencies : array
        Frequency band for the computation of the amplification factor
    mu : array
        Images magnifications
    td : array
        Images time delays
    n : array
        Morse indices

    Returns
    -------
    Fmag : array
        F(f) amplification factor
    """
    Fmag = np.zeros(len(frequencies), dtype=np.complex128)
    for i in range(len(mu)):
        # Frequency is a bin shorter than the NR waveform thus so will be Fmag
        # ref https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.utils.frequency_from_polarizations
        Fmag += np.sqrt(np.abs(mu[i])) * np.exp(1j*np.pi*(2.*frequencies*td[i] - n[i]))
    return Fmag


def geometricalOpticsMagnification(frequencies,
                                   Img_ra,
                                   Img_dec,
                                   source_pos_x,
                                   source_pos_y,
                                   zL,
                                   zS,
                                   lens_model_list,
                                   kwargs_lens_list,
                                   diff=None,
                                   scaled=False,
                                   scale_factor=None,
                                   cosmo=None):
    """
    Computes the geometrical optics amplification F(f) over a frequency band,
    given a set of images and a lens model

    Parameters
    ----------
    frequencies : array
        Frequency band for the computation of the amplification factor
    Img_ra : indexable object
        Images right ascensions (arbitrary units)
    Img_dec : indexable object
        Images declinations (arbitrary units)
    source_pos_x : float
        Source right ascension (arbitrary units)
    source_pos_y : float
        Source declination (arbitrary units)
    zL : float
        Lens redshift
    zS : float
        Source redshift
    lens_model_list : list of strings
        Names of the lens profiles to be considered for the lens model
    kwargs_lens_list : list of dictionaries
        Keyword arguments of the lens parameters matching each lens profile
        in lens_model_list
    diff : float, optional
        Step for numerical differentiation. Only needed for potentials that
        require numerical differentiation. If not specified, analytical
        differentiation is assumed
    scaled : bool, optional
        Specifies if the input is given in arbitrary units. If not specified,
        the input is assumed to be in radians
    scale_factor : float, optional
        Scale factor. Used to account for the proper conversion factor in the
        time delays when coordinates are given in arbitrary units, as per
        x_a.u. = x_radians/scale_factor. Only considered when scaled is True
    cosmo : instance of the astropy cosmology class, optional
        Cosmology used to compute angular diameter distances. If not specified,
        a FlatLambdaCDM instance with H_0=69.7, Omega_0=0.306, T_cmb0=2.725
        is considered

    Returns
    -------
    Fmag : array
        F(f) amplification factor
    """
    # Will store the results here
    td_list = []
    mu_list = []
    n_list = []
    Fmag = np.zeros(len(frequencies), dtype=np.complex128)

    # Time delays
    td_list = TimeDelay(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS,
                       lens_model_list, kwargs_lens_list, scaled=scaled,
                       scale_factor=scale_factor, cosmo=cosmo)

    # Magnifications
    mu_list = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list,
                           diff=diff)

    # Morse indices
    n_list = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list,
                           diff=diff)

    td = np.array(td_list)
    mu = np.array(mu_list)
    n = np.array(n_list)

    # Compute the amplification factor
    Fmag = amplification_from_data(frequencies, mu, td, n)

    return Fmag


class AmplificationCalculator:
    """
    Class for calculating various amplification effects in gravitational lensing.
    """
    
    def __init__(self, lens_model_list, kwargs_lens_list):
        """
        Initialize the amplification calculator.
        
        Parameters
        ----------
        lens_model_list : list
            List of lens model names
        kwargs_lens_list : list
            List of lens parameter dictionaries
        """
        self.lens_model_list = lens_model_list
        self.kwargs_lens_list = kwargs_lens_list
    
    def compute_frequency_amplification(self, frequencies, images_ra, images_dec,
                                      source_ra, source_dec, zL, zS):
        """
        Compute frequency-dependent amplification for given images.
        
        Parameters
        ----------
        frequencies : array
            Frequency array
        images_ra : array
            Image right ascensions
        images_dec : array
            Image declinations
        source_ra : float
            Source right ascension
        source_dec : float
            Source declination
        zL : float
            Lens redshift
        zS : float
            Source redshift
            
        Returns
        -------
        amplification : array
            Complex amplification factor
        """
        return geometricalOpticsMagnification(
            frequencies, images_ra, images_dec, source_ra, source_dec,
            zL, zS, self.lens_model_list, self.kwargs_lens_list
        )
