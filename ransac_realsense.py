import numpy as np
import matplotlib.pyplot as plt

ITERATION = 100

show = 1 # 0 = None, 1 = plot, 2 = ani

def get_point_cloud_from_file():

    pc = np.load('verts.npz')["verts"]

    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    for i in range(len(z)):
        if z[i] <= 0.0:
            x[i] = 0
            y[i] = 0
            z[i] = 0
        elif z[i] >= 6.0:
            x[i] = 0
            y[i] = 0
            z[i] = 0
    pc = np.column_stack((x[::10], y[::10], z[::10]))
    return pc
                
    
def plot(pc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pc[0][::20], pc[1][::20], pc[2][::20], color="gray")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.show()

def get_plane_vectors(point_cloud):
    # Generate three random indices
    random_indices = np.random.choice(len(point_cloud)-1, 3, replace=False)
    random_coordinates = point_cloud[random_indices]
    if random_coordinates[0][2] == 0 or random_coordinates[1][2] == 0 or random_coordinates[2][2] == 0:
        return get_plane_vectors(point_cloud)
    
    vector1 = random_coordinates[1] - random_coordinates[0]
    vector2 = random_coordinates[2] - random_coordinates[0]
    
    normal_vector = np.cross(vector1, vector2)
    
    return normal_vector, random_coordinates

def get_distance(point_cloud, normal_vector, threshold):
    distance_list = []
    z_val = [item[2] for item in point_cloud]
    for i in range(len(point_cloud)):
        if z_val[i] > 0:
            #print(z_val[i])
            distance = np.abs(np.divide(np.dot(point_cloud[i], normal_vector), np.linalg.norm(normal_vector)+0.00000000001))
        
            if distance < threshold:
                distance_list.append(distance)
            
    return len(distance_list)

def plot(point_cloud, plane_vectors, coords, ax):
    
    if show == 2:
        plt.ion()

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x[::20], y[::20], z[::20], color="gray")
    
    if plane_vectors is not None and coords is not None:
        point = coords[0]
        normal = plane_vectors

        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(5), range(5))

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /(normal[2]+0.0000000001)

        ax.plot_surface(xx, yy, zz, alpha=0.5, color='orange')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='blue')

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.title("3D Point Cloud with Plane")
    plt.grid(True)

    if show == 2:
        plt.pause(1.0)
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

    ax.scatter(x[::20], y[::20], z[::20])
    
    if plane_vectors is not None and coords is not None:
        point = coords[0]
        normal = plane_vectors

        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(5), range(5))

        # calculate corresponding z
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /(normal[2]+0.00000000001)

        ax.plot_surface(xx, yy, zz, alpha=0.5, color='green')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='red')

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    plt.title("3D Point Cloud with Best Fitted Plane")
    plt.grid(True)

    plt.show()

def main():
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = get_point_cloud_from_file()
    vec, coords = get_plane_vectors(pc)
    if show != 0:
        plot(pc, vec, coords, ax)
    
    plane_list = []
    
    for i in range(ITERATION):
        print("iteration: ", i)
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