#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency Analysis Example

This example demonstrates frequency domain analysis of lensed and unlensed
gravitational waves, including spectrograms and frequency evolution plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pycbc import waveform
from src.plotting.waveform_plotting import WaveformPlotter, plot_multiple_approximants


def frequency_evolution_analysis():
    """
    Analyze frequency evolution for different approximants.
    """
    print("Analyzing frequency evolution for different approximants...")
    
    # Parameters
    mass1, mass2 = 15.0, 15.0
    delta_t = 1.0/4096
    f_lower = 50.0
    
    approximants = ['IMRPhenomD', 'SEOBNRv4', 'SpinTaylorT4']
    
    # Plot frequency evolution comparison
    plot_multiple_approximants(
        approximants, 
        mass1=mass1, 
        mass2=mass2, 
        delta_t=delta_t, 
        f_lower=f_lower,
        plot_type='frequency'
    )


def pn_order_comparison():
    """
    Compare different Post-Newtonian orders for SpinTaylorT4.
    """
    print("Comparing Post-Newtonian orders...")
    
    plotter = WaveformPlotter()
    
    # Parameters
    approximant = 'SpinTaylorT4'
    mass1, mass2 = 30.0, 30.0
    delta_t = 1.0/4096
    f_lower = 50.0
    pn_orders = [2, 3, 4, 5, 6, 7]
    
    plotter.plot_pn_order_comparison(
        approximant, mass1, mass2, delta_t, f_lower, pn_orders
    )


def amplitude_phase_analysis():
    """
    Analyze amplitude vs phase relationships.
    """
    print("Analyzing amplitude vs phase relationships...")
    
    approximants = ['IMRPhenomD', 'SEOBNRv4', 'TaylorT4']
    plot_multiple_approximants(
        approximants,
        mass1=10, mass2=10,
        delta_t=1.0/4096,
        f_lower=40,
        plot_type='amplitude_phase'
    )


def mass_dependence_study():
    """
    Study the effect of different mass ratios on frequency evolution.
    """
    print("Studying mass dependence on frequency evolution...")
    
    plotter = WaveformPlotter()
    waveforms = {}
    
    # Different mass configurations
    mass_configs = [
        (10, 10, "Equal mass (10+10)"),
        (15, 15, "Equal mass (15+15)"),
        (20, 10, "Unequal mass (20+10)"),
        (30, 10, "High ratio (30+10)")
    ]
    
    delta_t = 1.0/4096
    f_lower = 50.0
    approximant = 'IMRPhenomD'
    
    for mass1, mass2, label in mass_configs:
        try:
            hp, hc = waveform.get_td_waveform(
                approximant=approximant,
                mass1=mass1,
                mass2=mass2,
                delta_t=delta_t,
                f_lower=f_lower
            )
            waveforms[label] = (hp, hc)
            
        except Exception as e:
            print(f"Could not generate waveform for {label}: {e}")
    
    plotter.plot_frequency_evolution(
        waveforms, 
        title="Frequency Evolution: Mass Dependence"
    )


def spin_effects_analysis():
    """
    Analyze the effects of spin on waveform characteristics.
    """
    print("Analyzing spin effects on waveforms...")
    
    plotter = WaveformPlotter()
    
    # Parameters
    mass1, mass2 = 20.0, 20.0
    delta_t = 1.0/4096
    f_lower = 50.0
    approximant = 'SEOBNRv4'
    
    # Different spin configurations
    spin_configs = [
        (0.0, 0.0, "No spin"),
        (0.5, 0.0, "Spin1 = 0.5"),
        (0.0, 0.5, "Spin2 = 0.5"),
        (0.9, 0.4, "Both spinning")
    ]
    
    time_series_data = {}
    
    for spin1z, spin2z, label in spin_configs:
        try:
            hp, hc = waveform.get_td_waveform(
                approximant=approximant,
                mass1=mass1,
                mass2=mass2,
                spin1z=spin1z,
                spin2z=spin2z,
                delta_t=delta_t,
                f_lower=f_lower
            )
            
            time_series_data[label] = (hp.sample_times, hp)
            
        except Exception as e:
            print(f"Could not generate waveform for {label}: {e}")
    
    plotter.plot_time_series(
        time_series_data,
        title="Spin Effects on Gravitational Waveforms",
        segment_indices=(len(hp.sample_times)//2, len(hp.sample_times))
    )


def frequency_domain_analysis():
    """
    Perform frequency domain analysis of waveforms.
    """
    print("Performing frequency domain analysis...")
    
    # Generate time domain waveform
    hp, hc = waveform.get_td_waveform(
        approximant='IMRPhenomD',
        mass1=30, mass2=30,
        delta_t=1.0/4096,
        f_lower=50
    )
    
    # Convert to frequency domain
    hp_fd = hp.to_frequencyseries()
    hc_fd = hc.to_frequencyseries()
    
    # Plot frequency domain amplitude
    plt.figure(figsize=(12, 8))
    
    frequencies = hp_fd.sample_frequencies
    amplitude = np.abs(hp_fd)
    
    plt.loglog(frequencies, amplitude, 'b-', linewidth=2, label='|h+|')
    plt.loglog(frequencies, np.abs(hc_fd), 'r--', linewidth=2, label='|hx|')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Strain Amplitude', fontsize=12)
    plt.title('Frequency Domain Gravitational Wave Amplitude', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 1000)
    plt.tight_layout()
    plt.show()
    
    # Plot phase
    plt.figure(figsize=(12, 6))
    
    phase_hp = np.unwrap(np.angle(hp_fd))
    phase_hc = np.unwrap(np.angle(hc_fd))
    
    plt.semilogx(frequencies, phase_hp, 'b-', linewidth=2, label='Phase(h+)')
    plt.semilogx(frequencies, phase_hc, 'r--', linewidth=2, label='Phase(hx)')
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Phase (radians)', fontsize=12)
    plt.title('Frequency Domain Gravitational Wave Phase', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 1000)
    plt.tight_layout()
    plt.show()


def chirp_mass_analysis():
    """
    Analyze the effect of chirp mass on frequency evolution.
    """
    print("Analyzing chirp mass effects...")
    
    plotter = WaveformPlotter()
    
    # Calculate chirp mass for different configurations
    def chirp_mass(m1, m2):
        return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    # Different total mass, same chirp mass
    mass_configs = [
        (20, 20, f"M_chirp = {chirp_mass(20, 20):.1f}"),
        (25, 15, f"M_chirp = {chirp_mass(25, 15):.1f}"),
        (30, 12, f"M_chirp = {chirp_mass(30, 12):.1f}"),
    ]
    
    waveforms = {}
    
    for mass1, mass2, label in mass_configs:
        try:
            hp, hc = waveform.get_td_waveform(
                approximant='IMRPhenomD',
                mass1=mass1,
                mass2=mass2,
                delta_t=1.0/4096,
                f_lower=50
            )
            waveforms[label] = (hp, hc)
            
        except Exception as e:
            print(f"Could not generate waveform for {label}: {e}")
    
    plotter.plot_frequency_evolution(
        waveforms,
        title="Frequency Evolution: Chirp Mass Comparison"
    )


def main():
    """
    Main function to run all frequency analysis examples.
    """
    print("Starting Frequency Analysis Examples")
    print("=" * 50)
    
    try:
        frequency_evolution_analysis()
    except Exception as e:
        print(f"Error in frequency evolution analysis: {e}")
    
    try:
        pn_order_comparison()
    except Exception as e:
        print(f"Error in PN order comparison: {e}")
    
    try:
        amplitude_phase_analysis()
    except Exception as e:
        print(f"Error in amplitude-phase analysis: {e}")
    
    try:
        mass_dependence_study()
    except Exception as e:
        print(f"Error in mass dependence study: {e}")
    
    try:
        spin_effects_analysis()
    except Exception as e:
        print(f"Error in spin effects analysis: {e}")
    
    try:
        frequency_domain_analysis()
    except Exception as e:
        print(f"Error in frequency domain analysis: {e}")
    
    try:
        chirp_mass_analysis()
    except Exception as e:
        print(f"Error in chirp mass analysis: {e}")
    
    print("\nFrequency analysis examples completed!")


if __name__ == "__main__":
    main()
