#!/usr/bin/env python3
"""
6D Pose Estimation - Main Entry Point
=====================================

This script serves as the main entry point for the 6D pose estimation system.
It provides a simple interface to run the pose estimation with proper path handling.
"""

import os
import sys

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

def main():
    """Main entry point for 6D pose estimation."""
    print("🚀 Starting 6D Pose Estimation System")
    print("=" * 50)
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please ensure the YOLOv8 model is placed in the models/ directory")
        return
    
    # Check if calibration file exists
    calib_path = os.path.join(os.path.dirname(__file__), 'calibration', 'charuco_intrinsics.npz')
    if not os.path.exists(calib_path):
        print("⚠️  Warning: Calibration file not found")
        print("Run calibration first: python calibration/calibration_charuco.py")
        print("Continuing with default camera parameters...")
    
    try:
        # Import and run pose estimation
        from main_pose import run_pose_estimation
        run_pose_estimation()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        print("Check camera connection and calibration")

if __name__ == "__main__":
    main() 