import numpy as np
import symnmf as sm 
import sys
from sklearn.metrics import silhouette_score
import math

EPSILON = 1e-4
DEFAULT_ERROR = "An Error Has Occurred"
MAX_ITER = 300

def calc_distance(point, centroid): 
    """
    Calculates the Euclidean distance between a data point and a centroid.
    """
    return math.dist(point, centroid) 

def choose_center(point, centers):
    """
    Chooses the closest centroid to a given data point using Euclidean distance.
    Returns the index of the closest centroid.
    """
    closest_center = (None, sys.float_info.max)
    for i in range(len(centers)):
        distance = calc_distance(point, centers[i]) 
        if distance < closest_center[1]:
            closest_center = (i, distance)
    return closest_center[0]

def data_process(data_file):
    """
    Processes the data file by reading each line, splitting it by commas, 
    converting each value to a float, and returning a list of data points (vectors).
    """
    points = []

    try:
        with open(data_file, 'r') as file:
            for line in file:
                linePoints = line.strip().split(',') 
                try:
                    linePoints = [float(i) for i in linePoints]  
                except ValueError:
                    print(DEFAULT_ERROR)
                    exit(1)            
                points.append(linePoints)
    except FileNotFoundError:
        print(DEFAULT_ERROR)
        exit(1)

    return points

def is_center_in_delta_threshold(point, center):
    """
    Checks if the distance between a point and a centroid is less than EPSILON.
    If the distance is less than EPSILON, the point is considered to be in the delta threshold.
    """
    if calc_distance(point, center) < EPSILON:
        return True
    return False

def is_centers_in_delta_threshold(curr_centers, prev_centers):
    """
    Checks if the centroids have converged by comparing their positions.
    If the maximum distance between the previous and current centroids is below EPSILON, they are considered converged.
    """
    for i in range(len(curr_centers)):
        if not is_center_in_delta_threshold(curr_centers[i], prev_centers[i]):
            return False
    return True

def calculate_center(pointsAroundCenter):
    """
    Calculates the centroid of a set of points by averaging their coordinates.
    """
    center = [0] * len(pointsAroundCenter[0])

    for point in pointsAroundCenter:
        for i in range(len(point)):
            center[i] += point[i]

    for i in range(len(center)):
        center[i] /= len(pointsAroundCenter)

    return center

def is_kmeans_arguments_valid(k, iter, points):
    return k > 0 and iter > 0 and len(points) > 0

def kmeans_alg(k, iter, points):
    """
    The main K-Means algorithm implementation.
    This function initializes cluster centers, iteratively assigns data points to clusters,
    and recalculates centroids until convergence or maximum iterations are reached.
    """
    if not is_kmeans_arguments_valid(k, iter, points):
        raise ValueError("Invalid arguments for K-Means algorithm")

    number_of_points = len(points)
    prev_chosen_centers = []
    
    # Initialize the centroids (initial cluster centers) by selecting the first k points
    chosen_centers = []
    for i in range(k):
        chosen_centers.append(points[i][:])

    iteration_number = 0
    is_centers_in_threshold = False

    while (iteration_number < iter and not is_centers_in_threshold):
        pointsAroundCenter = [[] for _ in range(k)]
        
        # Assign each point to the closest centroid
        for i in range(number_of_points):
            closest_center = choose_center(points[i], chosen_centers)
            pointsAroundCenter[closest_center].append(points[i])

        # copy the current centers to prev_choosen_centers
        prev_chosen_centers = [center[:] for center in chosen_centers]
        
        # Recalculate the centroids
        for i in range(k):
            chosen_centers[i] = calculate_center(pointsAroundCenter[i])

        is_centers_in_threshold = is_centers_in_delta_threshold(chosen_centers, prev_chosen_centers)
        iteration_number += 1

    return chosen_centers

def kmeans_sil(points, centers):
    clusters = [choose_center(point, centers) for point in points]
    return np.array(silhouette_score(points, clusters))

def symnmf_sil(points, H):
    # use H to assign to clusters
    clusters_assignment = np.argmax(H, axis=1) # extract cluster for each datapoint
    return silhouette_score(points, clusters_assignment)

def symnmf_sil_helper(points, k): 
    W = np.array(sm.norm(points))
    N = W.shape[0]
    m = np.mean(W)
    W = W.tolist()
    H0 = sm.initialize_H(N, k, m)
    return sm.symnmf(k, W, H0) #passing H

def main():
    args = sys.argv
    if len(args) != 3:
        print(DEFAULT_ERROR)
        exit(1)
    try:
        k, file_name = int(args[1]), args[2]
        kmeans_data = data_process(file_name)
        symnmf_data = data_process(file_name)

        symnmf_sil_result = symnmf_sil(symnmf_data, symnmf_sil_helper(symnmf_data, k))
        kmeans_sil_result = kmeans_sil(kmeans_data, kmeans_alg(k, MAX_ITER, kmeans_data))

        print("nmf:", "{:.4f}".format(symnmf_sil_result))
        print("kmeans:", "{:.4f}".format(kmeans_sil_result))
    except Exception as e:
        print(DEFAULT_ERROR)
        exit(1)

if __name__ == "__main__":
    main()