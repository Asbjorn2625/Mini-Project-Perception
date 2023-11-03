import numpy as np

# Create a sample 2D array
points = np.array([[1, 2, 0],
                 [4, 5, 6],
                 [7, 8, 0],
                 [10, 11, 12]])

# Compute the dot product for all points with the plane
dot_products = np.dot(points, np.transpose(points))

# Compute the norms of all points
point_norms = np.linalg.norm(points, axis=1)

# Compute distances for all points simultaneously
distances = np.where((points[:, 2] == 0) | (points[:, 2] < 0), 255, np.abs(dot_products / point_norms))
print(distances)
