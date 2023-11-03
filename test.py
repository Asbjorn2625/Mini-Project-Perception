import Libs.Cams as Cams
import Libs.Ransac as Ransac
import cv2

# Create a camera object
cam = Cams.RealsenseCamera()

# Initialize the camera
cam.initialize()

# Start capturing image
cam.start_capture()


