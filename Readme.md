# GW-Lensing-Analysis

A comprehensive Python package for gravitational wave lensing analysis using PyCBC and lenstronomy frameworks.

## Overview

This package provides tools for simulating and analyzing gravitational wave signals that have been gravitationally lensed by massive objects. It combines the power of PyCBC for gravitational wave generation and detection with lenstronomy for gravitational lensing calculations.

## Features

- **Lensed Waveform Generation**: Generate gravitational wave signals modified by gravitational lensing effects
- **Multiple Waveform Approximants**: Support for various approximants including IMRPhenomD, SEOBNRv4, SpinTaylorT4
- **Detector Response**: Calculate detector responses for LIGO (H1, L1), Virgo (V1), and other gravitational wave detectors
- **Lens Models**: Support for point mass lenses and other lens configurations
- **Visualization Tools**: Comprehensive plotting capabilities for waveforms, images, and detector responses
- **Time and Frequency Domain Analysis**: Tools for analyzing signals in both time and frequency domains

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- PyCBC
- lenstronomy

### Install from source

```bash
git clone https://github.com/yourusername/GW-Lensing-Analysis.git
cd GW-Lensing-Analysis
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Lensed Waveform Generation

```python
from src.core.lens_waveform_model import lens_waveform_model
from pycbc import waveform
import numpy as np

# Define lens parameters
source_ra = 1.2  # hours
source_dec = 15  # degrees
lens_ra = 0.1
lens_dec = 0.2
zs = 6.0  # source redshift
zl = 3.0  # lens redshift
ml = 1e6  # lens mass (solar masses)
lens_model_list = ['POINT_MASS']

# Binary parameters
mass1 = 30  # solar masses
mass2 = 30  # solar masses
delta_t = 1.0/4096
f_lower = 50  # Hz
distance = 6791.8106  # Mpc

# Generate lensed waveform
hp_lensed, hc_lensed = waveform.get_td_waveform(
    approximant="lensed",
    source_ra=source_ra, source_dec=source_dec,
    lens_ra=lens_ra, lens_dec=lens_dec,
    distance=distance,
    zs=zs, zl=zl, ml=ml,
    lens_model_list=lens_model_list,
    mass1=mass1, mass2=mass2,
    delta_t=delta_t, f_lower=f_lower
)
```

### Detector Response Analysis

```python
from pycbc.detector import Detector
from src.detectors.detector_response import calculate_detector_response

# Initialize detector
detector = Detector("H1")

# Calculate antenna pattern
fp, fc = detector.antenna_pattern(source_ra, source_dec, pol=0, time=0)

# Project waveform onto detector
strain = detector.project_wave(hp_lensed, hc_lensed, source_ra, source_dec, pol=0)
```

## Repository Structure

```
GW-Lensing-Analysis/
├── src/core/              # Core lensing algorithms
├── src/plotting/          # Visualization tools
├── src/detectors/         # Detector response calculations
├── examples/              # Example scripts
├── notebooks/             # Jupyter tutorials
├── tests/                 # Unit tests
├── config/                # Configuration files
└── docs/                  # Documentation
```

## Examples

The `examples/` directory contains several demonstration scripts:

- `basic_lensing_example.py`: Simple lensed waveform generation
- `frequency_analysis.py`: Time-frequency analysis of lensed signals

## Tutorials

Interactive Jupyter notebooks are provided in the `notebooks/` directory:

1. **Basic Lensing Tutorial**: Introduction to gravitational wave lensing
2. **Waveform Approximants Comparison**: Comparing different waveform models
3. **Detector Response Analysis**: Understanding detector responses
4. **Advanced Lensing Scenarios**: Complex lens configurations

## Supported Waveform Approximants

- **IMRPhenomD**: Inspiral-merger-ringdown phenomenological model
- **SEOBNRv4**: Spin-aligned effective-one-body model
- **SpinTaylorT4**: Spin-aligned post-Newtonian model
- **IMRPhenomB**: Non-spinning phenomenological model
- **TaylorF2**: Frequency-domain post-Newtonian model

## Supported Detectors

- **LIGO Hanford (H1)**
- **LIGO Livingston (L1)**
- **Virgo (V1)**
- **GEO600 (G1)**
- **KAGRA (K1)**

## Key Features

### Lensing Effects
- Time delays between multiple images
- Magnification of gravitational wave signals
- Frequency-dependent amplification
- Multiple image formation

### Analysis Tools
- Signal-to-noise ratio calculations
- Frequency evolution analysis
- Polarization studies
- Critical curve and caustic visualization

## Configuration

The package uses configuration files located in `config/waveform_config.ini` for default parameters. You can customize:

- Default waveform parameters
- Detector configurations
- Lens model parameters
- Analysis settings

## Testing

Run the test suite:

```bash
python -m pytest tests/
```




```bibtex
@software{gw_lensing_analysis,
  title={GW-Lensing-Analysis: Gravitational Wave Lensing Analysis Package},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/GW-Lensing-Analysis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyCBC**: For gravitational wave analysis tools
- **lenstronomy**: For gravitational lensing calculations
- **LIGO Scientific Collaboration**: For detector data and analysis methods
- **Einstein Toolkit**: For numerical relativity insights

## Support

For questions and support:
- Open an issue on GitHub
- Documentation: [Link to docs]



---

**Note**: This package is under active development. Some features may be experimental.
