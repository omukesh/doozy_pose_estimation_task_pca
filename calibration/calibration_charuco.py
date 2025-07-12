import cv2
import numpy as np
import pyrealsense2 as rs
import os
from cv2.aruco import CharucoBoard

# ------------------------
# Calibration Parameters
# ------------------------
SQUARES_X = 4
SQUARES_Y = 5
SQUARE_LENGTH = 0.039  # in meters
MARKER_LENGTH = 0.022  # in meters
NUM_SAMPLES_REQUIRED = 20
OUTPUT_FILE = "charuco_intrinsics.npz"

# ------------------------
# Charuco Board Setup
# ------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
charuco_board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)


# ------------------------
# RealSense Initialization
# ------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ------------------------
# Capture Loop
# ------------------------
print("[INFO] Starting capture... Press 's' to save sample, 'q' to quit.")

all_charuco_corners = []
all_charuco_ids = []
image_size = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

            # Interpolate Charuco corners
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco_board
            )

            if ret > 10:
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

        # Show the frame
        cv2.putText(image, f"Samples: {len(all_charuco_corners)}/{NUM_SAMPLES_REQUIRED}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Charuco Calibration", image)
        key = cv2.waitKey(1)

        if key == ord('s') and ids is not None and ret > 10:
            print("[INFO] Sample saved.")
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            image_size = gray.shape[::-1]

        elif key == ord('q') or len(all_charuco_corners) >= NUM_SAMPLES_REQUIRED:
            print("[INFO] Capture complete.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ------------------------
# Calibration
# ------------------------
if len(all_charuco_corners) < 5:
    print("[ERROR] Not enough samples for calibration.")
    exit()

print("[INFO] Running calibration...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=charuco_board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

# ------------------------
# Save Calibration
# ------------------------
np.savez(OUTPUT_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("[INFO] Calibration successful.")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print(f"[INFO] Saved to: {OUTPUT_FILE}")
