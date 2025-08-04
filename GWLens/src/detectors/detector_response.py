# -*- coding: utf-8 -*-
"""
Detector Response Module

This module provides functionality for calculating detector responses
to gravitational waves for various detectors.
"""

from pycbc.detector import Detector
import numpy as np


class DetectorResponse:
    """
    Class for handling detector responses to gravitational waves.
    """
    
    # Available detectors
    AVAILABLE_DETECTORS = ['H1', 'L1', 'V1', 'G1', 'K1']
    
    def __init__(self, detector_name='H1'):
        """
        Initialize detector response calculator.
        
        Parameters
        ----------
        detector_name : str
            Name of the detector ('H1', 'L1', 'V1', 'G1', 'K1')
        """
        if detector_name not in self.AVAILABLE_DETECTORS:
            raise ValueError(f"Detector {detector_name} not supported. "
                           f"Available: {self.AVAILABLE_DETECTORS}")
        
        self.detector_name = detector_name
        self.detector = Detector(detector_name)
    
    def antenna_pattern(self, right_ascension, declination, polarization, time):
        """
        Calculate antenna pattern coefficients.
        
        Parameters
        ----------
        right_ascension : float
            Right ascension in radians
        declination : float
            Declination in radians
        polarization : float
            Polarization angle in radians
        time : float
            GPS time
            
        Returns
        -------
        fp : float
            Plus polarization antenna pattern
        fc : float
            Cross polarization antenna pattern
        """
        return self.detector.antenna_pattern(right_ascension, declination,
                                           polarization, time)
    
    def project_wave(self, hp, hc, right_ascension, declination, polarization):
        """
        Project gravitational wave onto detector.
        
        Parameters
        ----------
        hp : TimeSeries
            Plus polarization
        hc : TimeSeries
            Cross polarization
        right_ascension : float
            Right ascension in radians
        declination : float
            Declination in radians
        polarization : float
            Polarization angle in radians
            
        Returns
        -------
        strain : TimeSeries
            Detector strain
        """
        return self.detector.project_wave(hp, hc, right_ascension, declination,
                                        polarization)
    
    def calculate_strain(self, hp, hc, right_ascension, declination,
                        polarization, time):
        """
        Calculate detector strain using antenna patterns.
        
        Parameters
        ----------
        hp : TimeSeries
            Plus polarization
        hc : TimeSeries
            Cross polarization
        right_ascension : float
            Right ascension in radians
        declination : float
            Declination in radians
        polarization : float
            Polarization angle in radians
        time : float
            GPS time
            
        Returns
        -------
        strain : TimeSeries
            Detector strain
        """
        fp, fc = self.antenna_pattern(right_ascension, declination,
                                    polarization, time)
        return fp * hp + fc * hc


class MultiDetectorResponse:
    """
    Class for handling multiple detector responses.
    """
    
    def __init__(self, detector_names=['H1', 'L1', 'V1']):
        """
        Initialize multi-detector response calculator.
        
        Parameters
        ----------
        detector_names : list
            List of detector names
        """
        self.detector_names = detector_names
        self.detectors = {name: DetectorResponse(name) for name in detector_names}
    
    def project_wave_all(self, hp, hc, right_ascension, declination, polarization):
        """
        Project gravitational wave onto all detectors.
        
        Parameters
        ----------
        hp : TimeSeries
            Plus polarization
        hc : TimeSeries
            Cross polarization
        right_ascension : float
            Right ascension in radians
        declination : float
            Declination in radians
        polarization : float
            Polarization angle in radians
            
        Returns
        -------
        strains : dict
            Dictionary of detector strains
        """
        strains = {}
        for name, detector in self.detectors.items():
            strains[name] = detector.project_wave(hp, hc, right_ascension,
                                                declination, polarization)
        return strains
    
    def compare_responses(self, hp, hc, right_ascension, declination, polarization):
        """
        Compare responses across multiple detectors.
        
        Parameters
        ----------
        hp : TimeSeries
            Plus polarization
        hc : TimeSeries
            Cross polarization
        right_ascension : float
            Right ascension in radians
        declination : float
            Declination in radians
        polarization : float
            Polarization angle in radians
            
        Returns
        -------
        comparison : dict
            Dictionary containing strain amplitudes and phases
        """
        strains = self.project_wave_all(hp, hc, right_ascension, declination,
                                      polarization)
        
        comparison = {}
        for name, strain in strains.items():
            comparison[name] = {
                'max_amplitude': np.max(np.abs(strain)),
                'strain': strain
            }
        
        return comparison


def setup_detector_network(detector_names=['H1', 'L1', 'V1']):
    """
    Set up a network of detectors.
    
    Parameters
    ----------
    detector_names : list
        List of detector names
        
    Returns
    -------
    network : MultiDetectorResponse
        Multi-detector response object
    """
    return MultiDetectorResponse(detector_names)
