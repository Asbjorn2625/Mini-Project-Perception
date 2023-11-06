# pipeline
# - Get point cloud
# - select 3 random coordinates from the point cloud
# - calculate the plane equation from the 3 coordinates
# - calculate the distance from the plane to all the points in the point cloud
# - if the distance is less than the threshold, add the point to the inlier list
# - iterate for n times
# - return the plane equation with the most inliers

import numpy as np
import matplotlib.pyplot as plt

ITERATION = 100

show = 2 # 0 = None, 1 = plot, 2 = ani

def get_point_cloud():
    # Define the number of random points and points on the plane
    num_random_points = 100
    num_plane_points = 100

    # Define the range for x, y, and z coordinates
    x_range = (0, 4)
    y_range = (0, 4)
    z_range = (0, 4)

    # Generate random x, y, and z coordinates for random points
    random_x = np.random.uniform(x_range[0], x_range[1], num_random_points)
    random_y = np.random.uniform(y_range[0], y_range[1], num_random_points)
    random_z = np.random.uniform(z_range[0], z_range[1], num_random_points)

    # Generate random x, y, and z coordinates for points on the plane
    plane_x = np.random.uniform(x_range[0], x_range[1], num_plane_points)
    plane_y = np.random.uniform(y_range[0], y_range[1], num_plane_points)
    plane_z = np.zeros(num_plane_points)  # Set z-coordinates to zero for the plane
    #plane_z = (2.3 * plane_x + 1.3 * plane_y + 2.5) * 1. /5.4

    # Combine the random points and points on the plane
    x_coordinates = np.concatenate([random_x, plane_x])
    y_coordinates = np.concatenate([random_y, plane_y])
    z_coordinates = np.concatenate([random_z, plane_z])

    # Create the 3D point cloud as a NumPy array
    point_cloud = np.column_stack((x_coordinates, y_coordinates, z_coordinates))

    print(point_cloud.shape)
    
    return point_cloud

def get_plane_vectors(point_cloud):
    # Generate three random indices
    random_indices = np.random.choice(len(point_cloud)-1, 3, replace=False)
    random_coordinates = point_cloud[random_indices]
    vector1 = random_coordinates[1] - random_coordinates[0]
    vector2 = random_coordinates[2] - random_coordinates[0]
    
    normal_vector = np.cross(vector1, vector2)
    
    return normal_vector, random_coordinates

def get_distance(point_cloud, normal_vector, threshold):
    distance_list = []
    for i in range(len(point_cloud)):
        distance = np.abs(np.divide(np.dot(point_cloud[i], normal_vector), np.linalg.norm(normal_vector)))
    
        if distance < threshold:
            distance_list.append(distance)
            
    return len(distance_list)

def plot(point_cloud, plane_vectors, coords, ax):
    
    if show == 2:
        plt.ion()

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, color="gray")
    
    if plane_vectors is not None and coords is not None:
        point = coords[0]
        normal = plane_vectors

        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(5), range(5))

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

        ax.plot_surface(xx, yy, zz, alpha=0.5, color='orange')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='blue')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.title("3D Point Cloud with Plane")
    plt.grid(True)

    if show == 2:
        plt.pause(0.1)
        plt.cla()
    elif show == 1:
        plt.show()

def plot_best_plane(point_cloud, plane_vectors, coords):
    
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z)
    
    if plane_vectors is not None and coords is not None:
        point = coords[0]
        normal = plane_vectors

        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(5), range(5))

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

        ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='red')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 4)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.title("3D Point Cloud with Best Fitted Plane")
    plt.grid(True)

    plt.show()

def main():
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = get_point_cloud()
    vec, coords = get_plane_vectors(pc)
    if show != 0:
        plot(pc, vec, coords, ax)
    
    plane_list = []
    
    for i in range(ITERATION):
        vec, coords = get_plane_vectors(pc)
        if show == 2:
            plot(pc, vec, coords, ax)
        inliers = get_distance(pc, vec, threshold=0.1)
        plane_list.append([inliers, vec, coords])
    
    inlier_list = [item[0] for item in plane_list]
    min_index = inlier_list.index(max(inlier_list))
    print("best plane: ", plane_list[min_index])
    
    plot_best_plane(pc, plane_list[min_index][1], plane_list[min_index][2])
if __name__ == "__main__":
    main()