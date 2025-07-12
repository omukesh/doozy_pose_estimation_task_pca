# 6D Pose Estimation with Intel RealSense Camera

This repository implements a real-time 6D pose estimation system using an Intel RealSense camera, YOLOv8 segmentation, and Principal Component Analysis (PCA) for robust pose estimation.

## 🎯 Overview

The system performs real-time 6D pose estimation (3D translation + 3D rotation) of objects using:
- **Intel RealSense D435i** camera for RGB-D data
- **YOLOv8 segmentation** for object detection and masking
- **Principal Component Analysis (PCA)** for orientation estimation
- **Advanced smoothing techniques** for stable pose outputs

## 📁 Repository Structure

```
doozy_task_pose_estimation/
├── models/
│   └── best.pt                 # YOLOv8 segmentation model
├── calibration/
│   └── calibration_charuco.py  # Camera calibration script
├── src/
│   ├── main_pose.py           # Main pose estimation script
│   ├── realsense1.py          # RealSense camera interface
│   └── inference.py           # YOLOv8 inference module
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Methodology

### 1. Camera Calibration
The system uses **ChArUco board calibration** for accurate camera intrinsics:

- **ChArUco Board**: 4×5 squares with 5×5 ArUco markers
- **Square Length**: 39mm, **Marker Length**: 22mm
- **Calibration Process**: 
  - Capture 20+ samples from different angles
  - Press 's' to save valid samples
  - Automatic camera matrix and distortion coefficient calculation
  - Results saved to `charuco_intrinsics.npz`

### 2. Object Detection & Segmentation
- **YOLOv8-seg**: Custom-trained segmentation model
- **Input**: RGB image from RealSense camera
- **Output**: Binary masks for detected objects
- **Mask Processing**: Erosion with 9×9 elliptical kernel to reduce edge noise

### 3. Centroid Stabilization
Multiple smoothing approaches for stable centroid estimation:

#### **Running Average Buffer**
```python
N = 10  # Window size
cx_buffer = deque(maxlen=N)
cy_buffer = deque(maxlen=N)
cx_avg = sum(cx_buffer) / len(cx_buffer)
```

#### **Exponential Smoothing**
```python
alpha_centroid = 0.5  # Smoothing factor
smoothed_cx = alpha_centroid * cx_avg + (1 - alpha_centroid) * smoothed_cx
```

#### **Median-based Centroid**
```python
cy, cx = np.median(mask_coords, axis=0).astype(int)
```

### 4. Depth Processing
- **Spatial Filtering**: Fill holes in depth data
- **Hole Filling**: Additional depth completion
- **Smart Sampling**: Use all valid mask pixels for depth
- **Outlier Rejection**: Filter depths between 100-3000mm

### 5. 3D Point Cloud Generation
```python
# Convert 2D pixels to 3D points
for y, x in mask_coords:
    z = depth_image[y, x]
    if valid_depth(z):
        Xp, Yp, Zp = XYZ_Cordinates(x, y, z * 0.001)
        points.append([Xp, Yp, Zp])
```

### 6. PCA-based Orientation Estimation
**Principal Component Analysis** for orientation estimation (with inherent limitations):

```python
# Calculate covariance matrix
mean = np.mean(points_3d, axis=0)
cov = np.cov(points_3d.T)

# Eigenvalue decomposition
eigvals, eigvecs = np.linalg.eig(cov)
idx = np.argsort(eigvals)[::-1]
rot_matrix = eigvecs[:, idx]

# Convert to Euler angles
roll, pitch, yaw = rotationMatrixToEulerAngles(rot_matrix)
```

**Note**: PCA-based orientation has accuracy limitations (typically ±5-15°) and may not be suitable for applications requiring high-precision orientation estimates. See "Accuracy Considerations" section for details.

### 7. Pose Smoothing
**Multi-level smoothing** for maximum stability:

#### **Translation Smoothing**
```python
# Running average + exponential smoothing
x_buffer.append(X)
X_avg = sum(x_buffer) / len(x_buffer)
smoothed_X = alpha * X_avg + (1 - alpha) * smoothed_X
```

#### **Rotation Smoothing**
```python
# Direct exponential smoothing on Euler angles
smoothed_roll = alpha * roll + (1 - alpha) * smoothed_roll
```

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd doozy_task_pose_estimation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Connect Intel RealSense camera** (D455 tested, D435i/D435 compatible)

## 📋 Usage

### Camera Calibration
```bash
cd calibration
python calibration_charuco.py
```
- Print a ChArUco board (4×5 squares)
- Hold board at different angles
- Press 's' to save samples (20+ recommended)
- Press 'q' to finish calibration

### Pose Estimation
```bash
cd src
python main_pose.py
```
- Real-time 6D pose estimation
- Press 'q' to quit
- Continuous detection mode enabled by default

## ⚙️ Parameters

### Smoothing Parameters
- `alpha = 0.1`: Translation smoothing factor
- `alpha_centroid = 0.5`: Centroid smoothing factor
- `N = 10`: Running average window size

### Processing Parameters
- **Erosion Kernel**: 9×9 elliptical
- **Depth Range**: 100-3000mm
- **Minimum Points**: 10 for PCA
- **Axis Length**: 50mm for visualization

## 📊 Output

The system provides real-time output:

```
6D Pose (PCA/Centroid, Smoothed):
Translation (X, Y, Z): 0.123, -0.045, 0.567 [meters]
Rotation (Roll, Pitch, Yaw): 12.34°, -5.67°, 89.12°
```

### Visualization
- **Centroid**: Yellow circle
- **Pose Axes**: RGB coordinate system
  - Red: X-axis
  - Green: Y-axis  
  - Blue: Z-axis
- **Mask Overlay**: Jet colormap
- **Text Overlay**: Real-time pose values

## 🔬 Technical Details

### Coordinate System
- **Camera Frame**: Right-handed coordinate system
- **X**: Right, **Y**: Down, **Z**: Forward
- **Units**: Meters for translation, degrees for rotation

### Performance
- **Frame Rate**: 30 FPS (RealSense default)
- **Resolution**: 848×480
- **Latency**: ~33ms per frame

### Camera Specifications
- **Tested Camera**: Intel RealSense D455
- **Minimum Depth Range**: 40cm (400mm)
- **Alternative Camera**: Intel RealSense D435 (minimum depth: 294mm)
- **Depth Tolerance**: ±2-5mm (camera hardware specification)

### Accuracy Considerations

#### **PCA-Based Orientation Limitations**
The system employs **Principal Component Analysis (PCA)** for orientation estimation, which has inherent limitations that affect accuracy:

**Mathematical Foundation**:
PCA computes the principal axes of the 3D point cloud by eigenvalue decomposition of the covariance matrix:
```python
cov = np.cov(points_3d.T)
eigvals, eigvecs = np.linalg.eig(cov)
rot_matrix = eigvecs[:, np.argsort(eigvals)[::-1]]
```

**Accuracy Limitations**:
1. **Geometric Assumptions**: PCA assumes the object's principal axes align with its geometric orientation, which may not hold for complex or asymmetric objects
2. **Point Cloud Quality**: Orientation accuracy depends on the quality and distribution of 3D points from the segmentation mask
3. **Symmetry Issues**: Objects with rotational symmetry (e.g., cylinders, spheres) may have ambiguous orientation estimates
4. **Noise Sensitivity**: PCA is sensitive to outliers and noise in the point cloud, affecting rotation matrix stability

**Expected Accuracy**:
- **Translation**: ±2-5mm (limited by RealSense depth tolerance)
- **Rotation**: ±5-15° (PCA-based, varies with object geometry)
- **Centroid**: ±1-3mm (smoothed, depends on segmentation quality)
- **Depth Range**: 40cm - 10m (D455), 29.4cm - 10m (D435)

**When PCA Works Well**:
- Objects with distinct geometric axes (rectangular prisms, elongated objects)
- High-quality depth data with minimal noise
- Objects with clear geometric features

**When PCA May Be Inaccurate**:
- Spherical or cylindrical objects (rotational ambiguity)
- Objects with complex, asymmetric geometry
- Poor depth data or segmentation quality
- Objects with similar dimensions along multiple axes

**Alternative Approaches**:
For applications requiring higher orientation accuracy, consider:
- Template-based pose estimation
- Deep learning pose regression networks
- Iterative Closest Point (ICP) algorithms
- CAD model-based pose refinement

### Stability Features
1. **Mask Erosion**: Reduces edge noise
2. **Temporal Smoothing**: Running averages + exponential smoothing
3. **Outlier Rejection**: Depth and coordinate filtering
4. **Median Sampling**: Robust centroid estimation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Intel RealSense SDK
- Ultralytics YOLOv8
- OpenCV community
- ChArUco calibration methodology

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review calibration quality
3. Verify camera connection
4. Open an issue with detailed logs 