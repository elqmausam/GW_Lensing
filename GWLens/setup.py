#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for GW-Lensing-Analysis package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gw-lensing-analysis",
    version="0.1.0",
    author="Saniya",
    description="A comprehensive toolkit for analyzing gravitational wave lensing effects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/GW-Lensing-Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gw-lensing=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.ini", "*.txt", "*.md"],
        "config": ["*.ini"],
    },
    zip_safe=False,
    keywords="gravitational waves, lensing, astronomy, physics, pycbc, ligo",
    project_urls={
        "Bug Reports": "https://github.com/elqmausam/GW_Lensing",
        "Source": "https://github.com/elqmausam/GW_Lensing",
        "Documentation": "https://gw-lensing-analysis.readthedocs.io/",
    },
)
