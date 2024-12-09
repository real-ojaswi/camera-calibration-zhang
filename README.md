# Camera Calibration and Homography Estimation

This repository contains code for camera calibration using Zhang's algorithm from scratch using geometric principles. The project involves computing the camera intrinsic and extrinsic parameters, refining the homographies using the RANSAC and Levenberg-Marquardt algorithms, and visualizing the results. The main components of the project include the following steps:

## Features:
1. **Edge Detection and Hough Transform**: 
   - Edge detection using Canny and Hough Transform to detect lines in the images of calibration pattern.
   - Grouping and merging detected lines based on angle and distance thresholds.
   - Detecting intersections of lines and labeling them.

2. **Camera Calibration**:
   - Estimation of camera homographies using world points and corresponding image points.
   - Intrinsic parameters are estimated from homographies using Singular Value Decomposition (SVD).
   - Extrinsic parameters (rotation and translation) are computed from the homographies.

3. **Optimization**:
   - Optimization of intrinsic and extrinsic parameters using Levenberg-Marquardt optimization.
   - Radial distortion correction is applied, and parameters are optimized.

4. **3D Visualization**:
   - The camera poses (extrinsic parameters) and the calibration pattern are visualized in 3D space.
   - Projected 3D points onto the image plane to check the accuracy of the homography estimation.
   - Camera frames and principal planes are drawn to show the camera orientation.

5. **Reprojection Error**:
   - Reprojection error is calculated for all images to measure the quality of the calibration.
   - Visualization of the reprojection of 3D points onto the images for validation.

6. **Homography Refinement**:
   - Homographies are refined using RANSAC to handle outliers, followed by Levenberg-Marquardt optimization.

## Key Functions:
- `detect_edges()`: Detects edges in an image using Canny edge detector.
- `hough_lines()`: Detects lines using the Probabilistic Hough Transform.
- `compute_homography()`: Computes the homography matrix using Direct Linear Transformation (DLT).
- `compute_intrinsic_params()`: Estimates the intrinsic camera parameters from homographies.
- `compute_extrinsic_params()`: Extracts rotation and translation from homographies.
- `optimize_parameters()`: Optimizes the camera parameters using Levenberg-Marquardt.
- `reprojection_error()`: Computes reprojection error between 3D world points and 2D image points.

## Dependencies:
- OpenCV
- NumPy
- Matplotlib
- SciPy
- scikit-learn

