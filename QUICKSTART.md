# Quick Start Guide

## Get Started in 5 Minutes

### Prerequisites
- Intel RealSense camera (D435i recommended)
- Python 3.8+
- USB 3.0 port

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Connect Camera
- Connect Intel RealSense camera via USB 3.0
- Ensure camera is recognized by your system

### 3. Run Pose Estimation
```bash
python main.py
```

That's it! The system will start real-time 6D pose estimation.

## Detailed Setup

### Camera Calibration (Recommended)
For best accuracy, calibrate your camera first:

```bash
cd calibration
python calibration_charuco.py
```

**Steps:**
1. Print a ChArUco board (4×5 squares)
2. Hold board at different angles
3. Press 's' to save samples (20+ recommended)
4. Press 'q' to finish

### Manual Execution
```bash
# From src directory
cd src
python main_pose.py
```

## Controls
- **q**: Quit application
- **s**: Save calibration sample (during calibration)

## Understanding Output

### Terminal Output
```
6D Pose (PCA/Centroid, Smoothed):
Translation (X, Y, Z): 0.123, -0.045, 0.567 [meters]
Rotation (Roll, Pitch, Yaw): 12.34°, -5.67°, 89.12°
```

### Visual Output
- **Yellow circle**: Object centroid
- **RGB axes**: Object orientation
  - Red: X-axis
  - Green: Y-axis
  - Blue: Z-axis
- **Colored overlay**: Segmentation mask

## 🔧 Troubleshooting

### Common Issues

**1. Camera not detected**
```bash
# Check camera connection
lsusb | grep Intel
```

**2. Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. Poor accuracy**
- Run camera calibration
- Ensure good lighting
- Check object distance (0.4-3m)

**4. Unstable pose**
- Adjust smoothing parameters in `main_pose.py`
- Increase erosion kernel size
- Check for camera movement

### Performance Tips
- Use USB 3.0 connection
- Ensure adequate lighting
- Keep objects within 0.1-3m range
- Minimize camera movement

## 📞 Need Help?

1. Check the main README.md
2. Review troubleshooting section
3. Open an issue with:
   - Error messages
   - System specifications
   - Camera model
   - Python version 