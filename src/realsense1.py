import pyrealsense2 as rs
import numpy as np

# Configure depth and color streams
print("Loading Intel Realsense Camera")
pipeline = rs.pipeline()

distorsion = np.array([-0.05892965,  0.05030509,  0.00244843,  0.00226107, 0.07880644])
cameraMatrix = np.array([
                            [520.7654115,   0.,         431.7113228],
                            [0.,        788.09755962, 242.33471377],
                        [0., 0., 1.]
])


config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# data to calculate x, y, z distances
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

# data for getting extrensic perameters from camara
stream_profile_color = profile.get_stream(rs.stream.color)
stream_profile_depth = profile.get_stream(rs.stream.depth)
color_intrs = stream_profile_color.as_video_stream_profile().get_intrinsics()
depth_intrs = stream_profile_depth.as_video_stream_profile().get_intrinsics()
extrinsics = stream_profile_depth.get_extrinsics_to(stream_profile_color)
print("extrins:\n", extrinsics)


def getFrames():
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        # If there is no frame, probably camera not connected, return False
        print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")

    # Apply filter to fill the Holes in the depth image
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    filtered_depth = spatial.process(depth_frame)

    hole_filling = rs.hole_filling_filter()
    filled_depth = hole_filling.process(filtered_depth)

    # Convert images to numpy arrays
    # distance = depth_frame.get_distance(int(50),int(50))
    # print("distance", distance)
    depth_image = np.asanyarray(filled_depth.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, depth_frame


def XYZ_Cordinates(pixelX, pixelY, depth):
    x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [pixelX, pixelY], depth)
    return x, y, z


def caliculateDepthFromPixelXY(depthFrame, cordinates):
    return depthFrame.get_distance(cordinates[0], cordinates[1])


def rotateAndTranslatePoint(rotation, translation):
    extrinsics = rs.pyrealsense2.extrinsics()
    extrinsics.rotation = rotation
    extrinsics.translation = translation
    return extrinsics


def translationOfPoint(cameraRotationTranslation, xyzPoint):
    return rs.rs2_transform_point_to_point(cameraRotationTranslation, xyzPoint)
