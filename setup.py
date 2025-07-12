#!/usr/bin/env python3
"""
Setup script for 6D Pose Estimation System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="6d-pose-estimation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time 6D pose estimation using Intel RealSense camera and YOLOv8",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/6d-pose-estimation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "6d-pose-estimation=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["models/*.pt", "calibration/*.py"],
    },
    keywords="pose-estimation, computer-vision, realsense, yolov8, pca, 6d-pose",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/6d-pose-estimation/issues",
        "Source": "https://github.com/yourusername/6d-pose-estimation",
        "Documentation": "https://github.com/yourusername/6d-pose-estimation#readme",
    },
) 