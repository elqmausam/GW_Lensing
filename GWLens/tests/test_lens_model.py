# -*- coding: utf-8 -*-
"""
Test module for lens waveform model functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.lens_waveform_model import LensWaveformModel
from src.core.utils import (
    eval_Einstein_radius, calculate_chirp_mass, calculate_mass_ratio,
    validate_redshifts, validate_mass, InvalidParameterError
)


class TestLensWaveformModel:
    """Test cases for LensWaveformModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LensWaveformModel()
        
        # Standard test parameters
        self.source_ra = 1.2
        self.source_dec = 15.0
        self.lens_ra = 0.1
        self.lens_dec = 0.2
        self.zs = 6.0
        self.zl = 3.0
        self.ml = 1e6
        self.lens_model_list = ['POINT_MASS']
        self.mass1 = 30.0
        self.mass2 = 30.0
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        assert isinstance(self.model, LensWaveformModel)
    
    def test_single_lens_eval_param(self):
        """Test parameter evaluation for single lens."""
        # Test with single lens mass
        ml_single = [self.ml]
        lens_ra_single = [self.lens_ra]
        lens_dec_single = [self.lens_dec]
        
        # Mock the microimages function since we don't have the full implementation
        with patch('src.core.lens_waveform_model.microimages') as mock_microimages:
            mock_microimages.return_value = (
                np.array([0.1, 0.2]), 
                np.array([0.3, 0.4]), 
                1e-10
            )
            
            result = self.model.eval_param(
                self.source_ra, self.source_dec, lens_ra_single, lens_dec_single,
                self.zs, self.zl, ml_single, self.lens_model_list, False
            )
            
            assert len(result) == 4  # Should return 4 elements
            assert mock_microimages.called
    
    def test_generate_lensed_waveform(self):
        """Test lensed waveform generation."""
        hp, hc = self.model.generate_lensed_waveform(
            source_ra=self.source_ra,
            source_dec=self.source_dec,
            lens_ra=self.lens_ra,
            lens_dec=self.lens_dec,
            zs=self.zs,
            zl=self.zl,
            ml=self.ml,
            lens_model_list=self.lens_model_list,
            mass1=self.mass1,
            mass2=self.mass2
        )
        
        # Check that waveforms are returned
        assert hp is not None
        assert hc is not None
        assert len(hp) > 0
        assert len(hc) > 0
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Test with negative mass
        with pytest.raises((ValueError, InvalidParameterError)):
            self.model.generate_lensed_waveform(
                source_ra=self.source_ra,
                source_dec=self.source_dec,
                lens_ra=self.lens_ra,
                lens_dec=self.lens_dec,
                zs=self.zs,
                zl=self.zl,
                ml=-1,  # Invalid negative mass
                lens_model_list=self.lens_model_list,
                mass1=self.mass1,
                mass2=self.mass2
            )


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_eval_Einstein_radius(self):
        """Test Einstein radius calculation."""
        zL, zS, mL = 0.5, 2.0, 1e12
        
        theta_E = eval_Einstein_radius(zL, zS, mL)
        
        assert isinstance(theta_E, float)
        assert theta_E > 0
        assert theta_E < 1  # Should be less than 1 radian
    
    def test_calculate_chirp_mass(self):
        """Test chirp mass calculation."""
        m1, m2 = 30.0, 20.0
        
        M_chirp = calculate_chirp_mass(m1, m2)
        expected = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        
        assert np.isclose(M_chirp, expected)
        assert M_chirp > 0
        assert M_chirp < m1 + m2
    
    def test_calculate_mass_ratio(self):
        """Test mass ratio calculation."""
        m1, m2 = 30.0, 20.0
        
        q = calculate_mass_ratio(m1, m2)
        
        assert 0 < q <= 1
        assert np.isclose(q, min(m1, m2) / max(m1, m2))
    
    def test_validate_redshifts_valid(self):
        """Test redshift validation with valid inputs."""
        # Should not raise any exception
        validate_redshifts(0.5, 2.0)
        validate_redshifts(0.0, 1.0)
        validate_redshifts(1.0, 1.1)
    
    def test_validate_redshifts_invalid(self):
        """Test redshift validation with invalid inputs."""
        # Negative redshifts
        with pytest.raises(InvalidParameterError):
            validate_redshifts(-0.1, 2.0)
        
        with pytest.raises(InvalidParameterError):
            validate_redshifts(0.5, -1.0)
        
        # Lens redshift >= source redshift
        with pytest.raises(InvalidParameterError):
            validate_redshifts(2.0, 1.0)
        
        with pytest.raises(InvalidParameterError):
            validate_redshifts(1.0, 1.0)
    
    def test_validate_mass_valid(self):
        """Test mass validation with valid inputs."""
        # Should not raise any exception
        validate_mass(1.0)
        validate_mass(100.0)
        validate_mass(1e12)
    
    def test_validate_mass_invalid(self):
        """Test mass validation with invalid inputs."""
        # Zero or negative mass
        with pytest.raises(InvalidParameterError):
            validate_mass(0)
        
        with pytest.raises(InvalidParameterError):
            validate_mass(-1.0)
