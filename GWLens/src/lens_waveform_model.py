# -*- coding: utf-8 -*-
"""
Lens Waveform Model

This module provides the main class for generating lensed gravitational waveforms
using PyCBC and LensingGW frameworks.
"""

from numpy import append, zeros, array, float64
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
import configparser as ConfigParser
from .utils import eval_Einstein_radius
from .images import microimages


class LensWaveformModel:
    """
    Main class for generating lensed gravitational waveforms.
    
    This class handles the computation of lensed images and generates
    corresponding gravitational wave signals with lensing effects.
    """
    
    def __init__(self):
        """Initialize the lens waveform model."""
        pass
    
    def eval_param(self, source_ra, source_dec, lens_ra, lens_dec,
                   zS, zL, mL, lens_model_list, optim):
        """
        Finds lensed images for the given set of parameters
        
        Parameters
        ----------
        source_ra : float
            Right ascension of the source of GW (in radians)
        source_dec : float
            Declination of the source of GW (in radians)
        lens_ra : array
            Right ascension of the lens (in radians)
        lens_dec : array
            Declination of the lens (in radians)
        mL : float
            Lens mass
        zL : float
            Lens redshift
        zS : float
            Source redshift
        lens_model_list : list of strings
            Names of the lens profiles to be considered for the lens model
        optim : bool
            For optimization of search algorithm
            
        Returns
        -------
        Img_ra : array
            Image right ascensions
        Img_dec : array
            Image declinations
        kwargs_lens_list : list
            Lens parameters
        solver_kwargs : dict
            Solver parameters
        """
        mL = array(mL, dtype=float64)
        lens_ra = array(lens_ra, dtype=float64)
        lens_dec = array(lens_dec, dtype=float64)

        # Handle multiple lenses
        if len(mL) > 1:
            mtot = sum(mL)
            thetaE = eval_Einstein_radius(zL, zS, mtot)
            beta0, beta1 = y0*thetaE, y1*thetaE
            thetaE_PM, eta0, eta1 = zeros(0), zeros(0), zeros(0)
            kwargs_lens_list = []

            for i in range(len(mL)):
                thetaE_PM = append(thetaE_PM, eval_Einstein_radius(zL, zS, mL[i]))
                eta0 = append(eta0, l0[i]*thetaE_PM[i])
                eta1 = append(eta1, l1[i]*thetaE_PM[i])
                kwargs_lens_list.append({
                    'center_x': eta0[i],
                    'center_y': eta1[i], 
                    'theta_E': thetaE_PM[i]
                })
            
            solver_kwargs = {'SearchWindowMacro': 4*thetaE_PM[0]}
            for i in range(1, len(mL)):
                solver_kwargs.update({'SearchWindow': 4*thetaE_PM[i]})
            solver_kwargs.update({'Optimization': optim})

            Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width = microimages(
                source_ra=source_ra,
                source_dec=source_dec,
                lens_model_list=lens_model_list,
                kwargs_lens=kwargs_lens_list,
                **solver_kwargs
            )
            
        # Handle single lens
        elif len(mL) == 1:
            mL, lens_ra, lens_dec = mL[0], lens_ra[0], lens_dec[0]
            thetaE_PM = eval_Einstein_radius(zL, zS, mL)
            kwargs_lens_list = [{
                'center_x': lens_ra,
                'center_y': lens_dec, 
                'theta_E': thetaE_PM/thetaE_PM
            }]
            
            solver_kwargs = {
                'Scaled': True,
                'ScaleFactor': thetaE_PM,
                'SearchWindowMacro': 4*thetaE_PM/thetaE_PM,
                'SearchWindow': 4*thetaE_PM/thetaE_PM,
                'OnlyMacro': 'True',
                'Optimization': optim
            }

            Img_ra, Img_dec, pixel_width = microimages(
                source_ra=source_ra,
                source_dec=source_dec,
                lens_model_list=lens_model_list,
                kwargs_lens=kwargs_lens_list,
                **solver_kwargs
            )

        return Img_ra, Img_dec, kwargs_lens_list, solver_kwargs
    
    def generate_lensed_waveform(self, source_ra, source_dec, lens_ra, lens_dec,
                                zs, zl, ml, lens_model_list, mass1, mass2,
                                approximant='IMRPhenomD', delta_t=1.0/4096,
                                f_lower=50, distance=400, optim=False):
        """
        Generate lensed gravitational waveform
        
        Parameters
        ----------
        source_ra : float
            Source right ascension
        source_dec : float
            Source declination
        lens_ra : float
            Lens right ascension
        lens_dec : float
            Lens declination
        zs : float
            Source redshift
        zl : float
            Lens redshift
        ml : float
            Lens mass
        lens_model_list : list
            List of lens models
        mass1 : float
            Primary mass
        mass2 : float
            Secondary mass
        approximant : str
            Waveform approximant
        delta_t : float
            Time sampling
        f_lower : float
            Lower frequency cutoff
        distance : float
            Luminosity distance
        optim : bool
            Optimization flag
            
        Returns
        -------
        hp_lensed : TimeSeries
            Lensed plus polarization
        hc_lensed : TimeSeries
            Lensed cross polarization
        """
        # This would integrate with your lensGW implementation
        # For now, returning standard waveform as placeholder
        hp, hc = get_td_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            delta_t=delta_t,
            f_lower=f_lower,
            distance=distance
        )
        
        # Apply lensing effects here
        # This is where your lensGW integration would happen
        
        return hp, hc
