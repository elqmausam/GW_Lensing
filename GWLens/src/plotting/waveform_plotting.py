# -*- coding: utf-8 -*-
"""
Waveform Plotting Module

This module provides plotting utilities for gravitational waveforms
and frequency analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pycbc import waveform


class WaveformPlotter:
    """
    Class for plotting gravitational waveforms and related quantities.
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn'):
        """
        Initialize the waveform plotter.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        style : str
            Matplotlib style
        """
        self.figsize = figsize
        plt.style.use('default')  # Use default style as seaborn might not be available
        
        # Set up plotting parameters
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        self.linestyles = ['-', '--', '-.', ':', '-', '--']
    
    def plot_time_series(self, time_series_dict, title="Gravitational Wave Time Series",
                        xlabel="Time (s)", ylabel="Strain", xlim=None, 
                        segment_indices=None):
        """
        Plot time series data.
        
        Parameters
        ----------
        time_series_dict : dict
            Dictionary with labels as keys and (times, values) tuples as values
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        xlim : tuple
            X-axis limits
        segment_indices : tuple
            (start_idx, end_idx) for plotting segments
        """
        plt.figure(figsize=self.figsize)
        
        for i, (label, (times, values)) in enumerate(time_series_dict.items()):
            color = self.colors[i % len(self.colors)]
            linestyle = self.linestyles[i % len(self.linestyles)]
            
            if segment_indices:
                start_idx, end_idx = segment_indices
                times = times[start_idx:end_idx]
                values = values[start_idx:end_idx]
            
            plt.plot(times, values, label=label, color=color, 
                    linestyle=linestyle, linewidth=1.5)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if xlim:
            plt.xlim(xlim)
        
        plt.tight_layout()
        plt.show()
    
    def plot_frequency_evolution(self, waveforms_dict, title="Frequency Evolution"):
        """
        Plot frequency evolution for different waveforms.
        
        Parameters
        ----------
        waveforms_dict : dict
            Dictionary with labels as keys and (hp, hc) tuples as values
        title : str
            Plot title
        """
        plt.figure(figsize=self.figsize)
        
        for i, (label, (hp, hc)) in enumerate(waveforms_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            # Trim zeros and calculate frequency
            hp_trimmed = hp.trim_zeros()
            hc_trimmed = hc.trim_zeros()
            
            freq = waveform.utils.frequency_from_polarizations(hp_trimmed, hc_trimmed)
            
            plt.plot(freq.sample_times, freq, label=label, color=color, linewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_amplitude_phase(self, waveforms_dict, title="Amplitude vs Phase"):
        """
        Plot amplitude vs phase for different waveforms.
        
        Parameters
        ----------
        waveforms_dict : dict
            Dictionary with labels as keys and (hp, hc) tuples as values
        title : str
            Plot title
        """
        plt.figure(figsize=self.figsize)
        
        for i, (label, (hp, hc)) in enumerate(waveforms_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            # Trim zeros
            hp_trimmed = hp.trim_zeros()
            hc_trimmed = hc.trim_zeros()
            
            # Calculate amplitude and phase
            amp = waveform.utils.amplitude_from_polarizations(hp_trimmed, hc_trimmed)
            phase = waveform.utils.phase_from_polarizations(hp_trimmed, hc_trimmed)
            
            plt.plot(phase, amp, label=label, color=color, linewidth=2)
        
        plt.xlabel('GW Phase (radians)', fontsize=12)
        plt.ylabel('GW Strain Amplitude', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_detector_comparison(self, detector_strains, segment_indices=None,
                               title="Detector Response Comparison"):
        """
        Plot comparison of detector responses.
        
        Parameters
        ----------
        detector_strains : dict
            Dictionary with detector names as keys and strain TimeSeries as values
        segment_indices : tuple
            (start_idx, end_idx) for plotting segments
        title : str
            Plot title
        """
        n_detectors = len(detector_strains)
        fig, axes = plt.subplots(n_detectors, 1, figsize=(12, 3*n_detectors))
        
        if n_detectors == 1:
            axes = [axes]
        
        for i, (det_name, strain) in enumerate(detector_strains.items()):
            color = self.colors[i % len(self.colors)]
            
            if segment_indices:
                start_idx, end_idx = segment_indices
                times = strain.sample_times[start_idx:end_idx]
                values = strain[start_idx:end_idx]
            else:
                times = strain.sample_times
                values = strain
            
            axes[i].plot(times, values, color=color, linewidth=1.5)
            axes[i].set_ylabel(f'{det_name} Strain', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(labelsize=10)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_pn_order_comparison(self, approximant, mass1, mass2, delta_t, f_lower,
                               pn_orders=[2, 3, 4, 5, 6, 7]):
        """
        Plot comparison of different PN orders for a given approximant.
        
        Parameters
        ----------
        approximant : str
            Waveform approximant
        mass1 : float
            Primary mass
        mass2 : float
            Secondary mass
        delta_t : float
            Time sampling
        f_lower : float
            Lower frequency cutoff
        pn_orders : list
            List of PN orders to compare
        """
        plt.figure(figsize=self.figsize)
        
        for i, phase_order in enumerate(pn_orders):
            try:
                hp, hc = waveform.get_td_waveform(
                    approximant=approximant,
                    mass1=mass1,
                    mass2=mass2,
                    phase_order=phase_order,
                    delta_t=delta_t,
                    f_lower=f_lower
                )
                
                hp_trimmed = hp.trim_zeros()
                hc_trimmed = hc.trim_zeros()
                freq = waveform.utils.frequency_from_polarizations(hp_trimmed, hc_trimmed)
                
                color = self.colors[i % len(self.colors)]
                plt.plot(freq.sample_times, freq, label=f"PN Order = {phase_order}",
                        color=color, linewidth=2)
                
            except Exception as e:
                print(f"Could not generate waveform for PN order {phase_order}: {e}")
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        plt.title(f'{approximant} - PN Order Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_multiple_approximants(approximants, mass1=15, mass2=15, delta_t=1.0/4096,
                             f_lower=50, plot_type='time_series'):
    """
    Utility function to plot multiple approximants.
    
    Parameters
    ----------
    approximants : list
        List of approximant names
    mass1 : float
        Primary mass
    mass2 : float
        Secondary mass
    delta_t : float
        Time sampling
    f_lower : float
        Lower frequency cutoff
    plot_type : str
        Type of plot ('time_series', 'frequency', 'amplitude_phase')
    """
    plotter = WaveformPlotter()
    waveforms = {}
    
    for approx in approximants:
        try:
            hp, hc = waveform.get_td_waveform(
                approximant=approx,
                mass1=mass1,
                mass2=mass2,
                delta_t=delta_t,
                f_lower=f_lower
            )
            waveforms[approx] = (hp, hc)
            
        except Exception as e:
            print(f"Could not generate waveform for {approx}: {e}")
    
    if plot_type == 'time_series':
        time_series = {}
        for label, (hp, hc) in waveforms.items():
            time_series[f'{label} (h+)'] = (hp.sample_times, hp)
        plotter.plot_time_series(time_series, title="Waveform Approximant Comparison")
        
    elif plot_type == 'frequency':
        plotter.plot_frequency_evolution(waveforms, title="Frequency Evolution Comparison")
        
    elif plot_type == 'amplitude_phase':
        plotter.plot_amplitude_phase(waveforms, title="Amplitude vs Phase Comparison")


# Example usage functions
def example_time_series_plot():
    """Example of plotting time series data."""
    from pycbc import waveform
    
    hp, hc = waveform.get_td_waveform(
        approximant='IMRPhenomD',
        mass1=30, mass2=30,
        delta_t=1.0/4096,
        f_lower=50
    )
    
    plotter = WaveformPlotter()
    time_series = {
        'h+ polarization': (hp.sample_times, hp),
        'hx polarization': (hc.sample_times, hc)
    }
    
    plotter.plot_time_series(time_series, title="IMRPhenomD Waveform")


def example_frequency_plot():
    """Example of plotting frequency evolution."""
    approximants = ['IMRPhenomD', 'SEOBNRv4']
    plot_multiple_approximants(approximants, plot_type='frequency')


if __name__ == "__main__":
    example_time_series_plot()
    example_frequency_plot()
