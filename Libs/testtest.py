import numpy as np
import os
import cv2
def resizeImage(image, scale_percent = 50):
    """Resize image to scale_percent% of original size"""
    # Calculate new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    
    # Resize image
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized
# Read image
folder = f"{os.getcwd()}\\data\\" 
intrinsic = np.load(folder + "intrinsics.npz")
sorted(intrinsic.files)
npzfile = np.load(folder + "verts1.npz")
sorted(npzfile.files)
verts = npzfile['verts']
texcoords = npzfile['texcoords']
coloursource = npzfile['color_source']
cv2.imwrite("Colour image.png", coloursource)
recolour = resizeImage(coloursource,50)

cv2.imshow("hej", recolour)
cv2.waitKey(0)