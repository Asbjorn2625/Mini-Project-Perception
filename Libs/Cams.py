import pyrealsense2 as rs
import numpy as np
import cv2
import os
import Ransac
import time
import cython

@cython.cdivision(True)
class RealsenseCamera:
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.curr_index = 0
        self.aligned_depth_frame = None
        self.color_image = None
        self.processing_thread = None

    def initialize(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a configuration for the pipeline
        config = rs.config()

        # Add the RGB and depth streams to the configuration
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

        # Start the pipeline
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # We will be removing the background of objects more than
        # clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1.5  # 1.5 meter
        self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        self.align_to = rs.stream.depth
        self.align = rs.align(self.align_to)

    def start_capture(self):
        try:
            while True:
                # Wait for the next set of frames from the camera
                frames = self.pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                self.aligned_depth_frame = aligned_frames.get_depth_frame()
                self.color_image = aligned_frames.get_color_frame()

                color_image = np.asanyarray(self.color_image.get_data())
                depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
                # Downsample the color image by keeping every second pixel
                downsampled_color_image = color_image[::2, ::2]

                # Downsample the depth image by keeping every second pixel
                downsampled_depth_image = depth_image[::2, ::2]
                self.process_data(downsampled_depth_image, downsampled_color_image)
                

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        finally:
            self.stop_capture()

    def stop_capture(self):
        # Stop the pipeline and release resources
        self.pipeline.stop()

    def process_data(self,depth_image, color_image):
        if self.aligned_depth_frame is not None:
            # use the ransec algorithm to find the plane
            # get the points from the depth image
        
            points = self._gather_points(depth_image)
            
            Ransacy = Ransac.Ransacy(points, 20, 100)

            zero_count = np.count_nonzero(depth_image == 0)
            non_zero_count = np.count_nonzero(depth_image != 0)
            total_count = zero_count + non_zero_count

            if total_count > 0:
                zero_ratio = zero_count / total_count
                non_zero_ratio = non_zero_count / total_count
            else:
                zero_ratio = 0.0
                non_zero_ratio = 0.0

            print("Ratio of zero values to total values:", zero_ratio)
            print("Ratio of non-zero values to total values:", non_zero_ratio)
            best_plane, best_inliers = Ransacy.find_best_plane()
            inlier_mask = np.zeros(depth_image.shape, dtype=np.uint8)
            inlier_mask = self._mark_inliers(inlier_mask, best_inliers)
           
            # Apply the mask to the colour image
            masked = cv2.bitwise_and(color_image, color_image, mask=inlier_mask)
            # Dialate the mask
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.dilate(masked, kernel, iterations=2)     
            self._resize_image(masked, "Ransac", 2)    


    def _gather_points(self, depth_image):
        height, width = depth_image.shape
        point_list = []
        for i in range(width):
            for j in range(height):
                x = i
                y = j
                z = depth_image[j,i] 
                point_list.append([x, y, z])
            
        return np.array(point_list)
        
    def _mark_inliers(self, inlier_mask, best_inliers):
        if len(best_inliers) == 0:
            return inlier_mask
        # Mark the inlier pixels in the mask using NumPy
        best_inliers= np.array(best_inliers.tolist())
        i_values = best_inliers[:, 1].astype(int)
        j_values = best_inliers[:, 0].astype(int)
        inlier_mask[i_values, j_values] = 255  # You can use any value to mark the inliers (e.g., 255 for white)
        return inlier_mask



    def _resize_image(self, image, image_name, percent):
        [height, width] = [image.shape[0], image.shape[1]]
        [height, width] = [percent * height, percent * width]
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_name, int(width), int(height))
        cv2.imshow(image_name, image)

if __name__ == "__main__":
    camera = RealsenseCamera()
    camera.initialize()
    camera.start_capture()
    