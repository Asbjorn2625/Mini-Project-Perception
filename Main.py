import Libs.Ransac as Ransac
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def resizeImage(image, scale_percent = 50):
    """Resize image to scale_percent% of original size"""
    # Calculate new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    
    # Resize image
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    
    return resized
def remove_non_green_verts(verts, texcoords, color_source):
    verts = verts.tolist()
    Hsvimage = cv2.cvtColor(color_source, cv2.COLOR_BGR2HSV)
    np.clip(texcoords, 0, 1, out=texcoords)
    for i in tqdm(range(len(verts))):
        x_color = (texcoords[i, 0] * (color_source.shape[0] - 1)).round().astype(int)
        y_color = (texcoords[i, 1] * (color_source.shape[1] - 1)).round().astype(int)
        
        tmp = Hsvimage[x_color,y_color]
        if (tmp[0] > 35 and tmp[0] < 85) and (tmp[1] > 25 and tmp[1] < 255) and (tmp[2] > 25 and tmp[2] < 255):
            verts[i] = [0,0,0]
    green_verts = np.array(verts)
    return green_verts

def plot_inliers_on_color(index, texcoords, color_source):
    #Here we take the index and finde
    x_color = []
    y_color = []
    for _, indices in enumerate(index):
        x_color.append((texcoords[indices, 0] * (color_source.shape[0] - 1)).round().astype(int))
        y_color.append((texcoords[indices, 1] * (color_source.shape[1] - 1)).round().astype(int))
    mask = np.zeros((color_source.shape[0], color_source.shape[1]), dtype=np.uint8) #NOTE ER IKKE SIKKER PÅ RÆKKEFØLGEN
    # Set the specified coordinates to red (assuming BGR color format)
    
    mask[np.array(x_color), np.array(y_color)] = 1
    #dilate mask
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 5)

    
    #Bitwise and with original image
    res = cv2.bitwise_and(color_source, color_source, mask=mask)
    return res

# Read image
folder = f"{os.getcwd()}\\data\\" 
intrinsic = np.load(folder + "intrinsics.npz")
sorted(intrinsic.files)

# Read image
folder = f"{os.getcwd()}\\data\\" 
npzfile = np.load(folder + "verts1.npz")
sorted(npzfile.files)
verts = npzfile['verts']
texcoords = npzfile['texcoords']
coloursource = npzfile['color_source']
#print(np.min(texcoords))
#print(np.max(texcoords))

#plot uv cords
#plt.scatter(texcoords[:,0], texcoords[:,1], s=0.1)
#plt.show()





# Set points with x,y or z values greater than 5 to 0

verts[np.abs(verts[:, 0]) > 5] = 0
verts[np.abs(verts[:, 1]) > 5] = 0
verts[np.abs(verts[:, 2]) > 5] = 0


# Resize image
recoloursource = resizeImage(coloursource, 50)
#print((coloursource.shape[0] - 1))
cv2.imshow("image", recoloursource)
# Remove non green verts
green_verts = remove_non_green_verts(verts, texcoords, coloursource)

Ransac = Ransac.RansacClass(green_verts, 5, 800)
bestinliers, indices = Ransac.find_best_plane()

#WE TEST PLOT INLIERS ON COLOR
inlierimg = plot_inliers_on_color(indices, texcoords, coloursource)
inlierimg = resizeImage(inlierimg, 50)
cv2.imshow("image", inlierimg)

# Plot the data
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
# Set the view angle to look directly into the z-axis
ax1.view_init(elev=90, azim=0)
# Add labels or customize other plot properties as needed
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
ax1.scatter(verts[::10,0], verts[::10,1], verts[::10,2], c='r', marker='o')
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(bestinliers[::10,0], bestinliers[::10,1], bestinliers[::10,2], c='y', marker='o')
ax1.legend()
ax2.legend()
plt.show()

"""
# Plot the data
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
# Set the view angle to look directly into the z-axis
ax1.view_init(elev=90, azim=0)
# Add labels or customize other plot properties as needed
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
ax1.scatter(verts[::10,0], verts[::10,1], verts[::10,2], c='r', marker='o')

#Run ransac on npzfile
BestPlaneList = Ransac.RansacClass(verts, 0.1, 50)
best_plane = BestPlaneList.find_best_plane()
#print(best_plane)
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(best_plane[::10,0], best_plane[::10,1], best_plane[::10,2], c='y', marker='o')
ax1.legend()
ax2.legend()
plt.show()
"""
#verts.shape = texcoords.shape
#print(len(verts))
#depth.shape = 1280X720=921600
#rgb = 1920*1080= 2073600







def plotplane(verts, plane):
    """draw plane"""
    
    # Get the plane normal and point on plane
    

    return 0



