# main_pose.py
import cv2
import numpy as np
import math
from typing import Optional
from realsense1 import getFrames, XYZ_Cordinates, cameraMatrix, distorsion
from inference import detect_and_segment
from collections import deque


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])  # roll
        y = math.atan2(-R[2, 0], sy)      # pitch
        z = math.atan2(R[1, 0], R[0, 0])  # yaw
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])  # Convert to degrees


def get_mask_centroid(mask):
    coords = np.column_stack(np.where(mask > 0.5))  # [y, x]
    if coords.shape[0] == 0:
        return None
    centroid = coords.mean(axis=0).astype(int)
    return (centroid[1], centroid[0])  # (x, y)


def run_pose_estimation():
    print("6D pose detection running. Press 'q' to quit.")

    # Smoothing variables
    smoothed_X: Optional[float] = None
    smoothed_Y: Optional[float] = None
    smoothed_Z: Optional[float] = None
    smoothed_roll: Optional[float] = None
    smoothed_pitch: Optional[float] = None
    smoothed_yaw: Optional[float] = None
    alpha = 0.1  # Smoothing factor

    # Temporal smoothing for centroid
    smoothed_cx: Optional[float] = None
    smoothed_cy: Optional[float] = None
    alpha_centroid = 0.5 # Smoothing factor for centroid

    N = 10  # window size
    x_buffer = deque(maxlen=N)
    y_buffer = deque(maxlen=N)
    z_buffer = deque(maxlen=N)
    cx_buffer = deque(maxlen=N)
    cy_buffer = deque(maxlen=N)

    while True:
        color_image, depth_image, depth_frame = getFrames()
        if color_image is None:
            continue

        cv2.imshow("Live Feed", color_image)
        key = cv2.waitKey(1) & 0xFF

        # Convert BGR to RGB for detection
        #rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        detections = detect_and_segment(color_image)
        for det in detections:
            mask = cv2.resize(det['mask'], (color_image.shape[1], color_image.shape[0]))
            # Erode the mask to use only the central region and avoid edge noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            # Use median of all valid mask pixels as centroid for stability
            mask_coords = np.column_stack(np.where(mask > 0.5))
            if mask_coords.shape[0] == 0:
                continue
            cy, cx = np.median(mask_coords, axis=0).astype(int)

            # Temporal smoothing for centroid (now with running average)
            cx_buffer.append(cx)
            cy_buffer.append(cy)
            cx_avg = sum(cx_buffer) / len(cx_buffer)
            cy_avg = sum(cy_buffer) / len(cy_buffer)
            if smoothed_cx is None or smoothed_cy is None:
                smoothed_cx, smoothed_cy = cx_avg, cy_avg
            else:
                smoothed_cx = alpha_centroid * cx_avg + (1 - alpha_centroid) * smoothed_cx
                smoothed_cy = alpha_centroid * cy_avg + (1 - alpha_centroid) * smoothed_cy
            cx_used = int(round(smoothed_cx))
            cy_used = int(round(smoothed_cy))

            # Smarter depth sampling (use all valid mask depths)
            mask_depths = [depth_image[y, x] for y, x in mask_coords if 100 < depth_image[y, x] < 3000]
            if len(mask_depths) == 0:
                print("No valid depth in mask.")
                continue
            depth = np.median(mask_depths) * 0.001  # mm to meters

            # Get 3D point of fixed centroid
            X, Y, Z = XYZ_Cordinates(cx_used, cy_used, depth)

            # Running average for X, Y, Z before smoothing
            x_buffer.append(X)
            y_buffer.append(Y)
            z_buffer.append(Z)
            X_avg = sum(x_buffer) / len(x_buffer)
            Y_avg = sum(y_buffer) / len(y_buffer)
            Z_avg = sum(z_buffer) / len(z_buffer)

            # Get 3D points for PCA (use all valid mask pixels)
            points = []
            for y, x in mask_coords:
                z = depth_image[y, x]
                if z == 0 or z < 100 or z > 3000:
                    continue
                Xp, Yp, Zp = XYZ_Cordinates(x, y, z * 0.001)
                points.append([float(Xp), float(Yp), float(Zp)])
            points_3d = np.array(points)
            if len(points_3d) < 10:
                print("Not enough 3D points for PCA.")
                continue

            # PCA
            mean = np.mean(points_3d, axis=0)
            cov = np.cov(points_3d.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            idx = np.argsort(eigvals)[::-1]
            rot_matrix = eigvecs[:, idx]

            # Euler angles
            roll, pitch, yaw = rotationMatrixToEulerAngles(rot_matrix)

            # Exponential smoothing for translation and rotation
            if smoothed_X is None or smoothed_Y is None or smoothed_Z is None or \
               smoothed_roll is None or smoothed_pitch is None or smoothed_yaw is None:
                smoothed_X, smoothed_Y, smoothed_Z = X_avg, Y_avg, Z_avg
                smoothed_roll, smoothed_pitch, smoothed_yaw = roll, pitch, yaw
            else:
                smoothed_X = alpha * X_avg + (1 - alpha) * smoothed_X
                smoothed_Y = alpha * Y_avg + (1 - alpha) * smoothed_Y
                smoothed_Z = alpha * Z_avg + (1 - alpha) * smoothed_Z
                smoothed_roll = alpha * roll + (1 - alpha) * smoothed_roll
                smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch
                smoothed_yaw = alpha * yaw + (1 - alpha) * smoothed_yaw

            # Print 6D pose in terminal (smoothed)
            print("\n6D Pose (PCA/Centroid, Smoothed):")
            print(f"Translation (X, Y, Z): {smoothed_X:.3f}, {smoothed_Y:.3f}, {smoothed_Z:.3f} [meters]")
            print(f"Rotation (Roll, Pitch, Yaw): {smoothed_roll:.2f}°, {smoothed_pitch:.2f}°, {smoothed_yaw:.2f}°")

            # Draw fixed centroid
            cv2.circle(color_image, (cx_used, cy_used), 6, (0, 255, 255), -1)

            # Draw pose axes on centroid (manual)
            axis_length = 0.05  # meters
            axis_3D = np.array([
                [0.0, 0.0, 0.0],
                [axis_length, 0.0, 0.0],
                [0.0, axis_length, 0.0],
                [0.0, 0.0, axis_length]
            ], dtype=np.float32).reshape(-1, 3)

            rvec, _ = cv2.Rodrigues(rot_matrix)
            tvec = np.array([[X], [Y], [Z]])

            imgpts, _ = cv2.projectPoints(axis_3D, rvec, tvec, cameraMatrix, distorsion)
            imgpts = imgpts.reshape(-1, 2).astype(int)

            origin = (cx_used, cy_used)
            cv2.line(color_image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X - Red
            cv2.line(color_image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y - Green
            cv2.line(color_image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z - Blue

            # Text overlay (smoothed values)
            cv2.putText(color_image, f"X: {smoothed_X:.2f}m Y: {smoothed_Y:.2f}m Z: {smoothed_Z:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(color_image, f"Roll: {smoothed_roll:.1f}* Pitch: {smoothed_pitch:.1f}* Yaw: {smoothed_yaw:.1f}*",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            # Mask overlay
            mask_colored = cv2.applyColorMap((mask * 255).astype('uint8'), cv2.COLORMAP_JET)
            color_image = cv2.addWeighted(color_image, 1, mask_colored, 0.5, 0)

            # Show final output in RGB
            rgb_display = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("6D Pose Estimation", rgb_display)

        
        # if key == ord('d'):
        #     detections = detect_and_segment(color_image)
        #     for det in detections:
        #         ... (copy the detection code block from above here)

        if key == ord('q'):
            print("Exiting.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_estimation()
