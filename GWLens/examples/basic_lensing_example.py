#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Lensing Example

This example demonstrates how to generate a lensed gravitational waveform
and compare it with an unlensed signal across multiple detectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform
from src.detectors.detector_response import DetectorResponse, MultiDetectorResponse
from src.core.lens_waveform_model import LensWaveformModel


def main():
    """
    Main function demonstrating basic lensing analysis.
    """
    # Physical parameters
    source_ra = 1.2  # hours
    source_dec = 15.0  # degrees
    lens_ra = 0.1
    lens_dec = 0.2
    zs = 6.0  # source redshift
    zl = 3.0  # lens redshift
    ml = 1e6  # lens mass
    lens_model_list = ['POINT_MASS']
    
    # Binary parameters
    mass1 = 30.0  # solar masses
    mass2 = 30.0  # solar masses
    distance = 400.0  # Mpc
    
    # Waveform parameters
    delta_t = 1.0/4096
    f_lower = 50.0
    approximant = 'IMRPhenomD'
    
    # Sky location and orientation
    pol = 0.2  # polarization angle
    inc = 0.0  # inclination
    time = -0.05  # merger time
    
    print("Generating gravitational waveforms...")
    
    # Generate unlensed waveform
    hp_unlensed, hc_unlensed = waveform.get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        f_lower=f_lower,
        delta_t=delta_t,
        inclination=inc,
        distance=distance
    )
    
    # Set start time
    hp_unlensed.start_time = hc_unlensed.start_time = time
    
    # For demonstration, we'll use the same waveform as "lensed"
    # In practice, this would come from your lensGW implementation
    hp_lensed = hp_unlensed.copy()
    hc_lensed = hc_unlensed.copy()
    
    print("Calculating detector responses...")
    
    # Set up detectors
    detector_network = MultiDetectorResponse(['H1', 'L1', 'V1'])
    
    # Calculate detector responses
    strains_unlensed = detector_network.project_wave_all(
        hp_unlensed, hc_unlensed, source_ra, source_dec, pol
    )
    
    strains_lensed = detector_network.project_wave_all(
        hp_lensed, hc_lensed, source_ra, source_dec, pol
    )
    
    print("Creating plots...")
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    detectors = ['H1', 'L1', 'V1']
    colors = ['red', 'blue', 'green']
    
    for i, (det, color) in enumerate(zip(detectors, colors)):
        # Plot short segment for clarity
        start_idx = 8591
        end_idx = 9000
        
        strain_u = strains_unlensed[det]
        strain_l = strains_lensed[det]
        
        time_segment = strain_u.sample_times[start_idx:end_idx]
        strain_u_segment = strain_u[start_idx:end_idx]
        strain_l_segment = strain_l[start_idx:end_idx]
        
        axes[i].plot(time_segment, strain_u_segment, 
                    label=f'Unlensed {det}', color=color, alpha=0.7)
        axes[i].plot(time_segment, strain_l_segment, 
                    label=f'Lensed {det}', color=color, linestyle='--')
        
        axes[i].set_ylabel(f'{det} Strain')
        axes[i].legend()
        axes[i].grid(True)
    
    axes[-1].set_xlabel('Time [s]')
    plt.suptitle('Gravitational Wave Strain: Lensed vs Unlensed')
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nDetector Response Statistics:")
    print("-" * 40)
    for det in detectors:
        strain_u = strains_unlensed[det]
        strain_l = strains_lensed[det]
        
        max_amp_u = np.max(np.abs(strain_u))
        max_amp_l = np.max(np.abs(strain_l))
        
        print(f"{det}:")
        print(f"  Unlensed max amplitude: {max_amp_u:.2e}")
        print(f"  Lensed max amplitude: {max_amp_l:.2e}")
        print(f"  Ratio (L/U): {max_amp_l/max_amp_u:.3f}")
        print()


def compare_approximants():
    """
    Compare different waveform approximants.
    """
    approximants = ['IMRPhenomD', 'SEOBNRv4', 'SpinTaylorT4']
    mass1, mass2 = 30.0, 30.0
    delta_t = 1.0/4096
    f_lower = 50.0
    
    plt.figure(figsize=(12, 8))
    
    for approx in approximants:
        try:
            hp, hc = waveform.get_td_w
