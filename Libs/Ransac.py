
import numpy as np
import random
import cython

@cython.cdivision(True)
class RansacClass:
    def __init__(self, points, threshold, iterations):
        self.points = points
        self.propoints = self.preprocess(self.points)
        self.threshold = threshold
        self.iterations = iterations
    def preprocess(self,points1):
        # Filter rows where the element in the 3rd column is not 0
        filtered_data = points1[points1[:, 2] != 0]
        # Sort the filtered data by the 3rd column
        points1 = filtered_data[filtered_data[:, 2].argsort()]
        return points1
    def find_best_plane(self):
        best_plane = None
        best_inliers = []
        for _ in range(self.iterations):
            random_indices = random.sample(range(len(self.propoints)), 3)
            random_points = self.points[random_indices]
                    
            
            plane = np.cross(random_points[1] - random_points[0], random_points[2] - random_points[0])

            plane_norm=np.linalg.norm(plane)

            distances = []
            for i in range(len(self.points)):
                if self.points[i][2] <= 0:
                    distances.append(self.threshold+10) # max, will never be conisdered
                else:
                    distances.append(np.abs(np.divide(np.dot(self.points[i], plane), plane_norm)))
            distances = np.array(distances) 
            inlier_indices = np.where(distances <= self.threshold)
            inlier_mask = np.where(distances <= self.threshold, 1,0 )
            vert_with_inliers = self.points & inlier_mask
            inliers = self.points[inlier_indices]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_massk = vert_with_inliers
 
        return best_inliers, best_massk