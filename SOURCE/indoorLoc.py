import numpy as np
import math
from scipy.spatial import distance


# ----------------------------------
# IndoorLoc class
# Get the location of a set of samples using a knn-based algorithm
# ----------------------------------
class IndoorLoc:
    # ---------------------------------------------------------
    # Class constructor
    # train_fingerprints is a np array of n_train_samples x n_aps.
    #    Each element i,j is the RSSI in the i-th location from the j-th AP
    # train_locations is a np array of train_samples x 2.
    #    Each element i is the coordinates x,y of i-th train fingerprint
    # k is the number of neighbors in the knn algorithm
    # ---------------------------------------------------------
    def __init__(self, train_fingerprints, train_locations, k = 3):
        self.train_fingerprints = train_fingerprints
        self.train_locations = train_locations
        self.n_training_samples = train_fingerprints.shape[0]
        self.n_aps = train_fingerprints.shape[1]
        self.n_floors = int(np.max(self.train_locations[:, 2]) + 1)
        self.k = k

    # ---------------------------------------------------------
    # get_floor
    # ---------------------------------------------------------
    # Convert the floor id in z coordinates assuming that the distance between two floors is 4 meters
    # ---------------------------------------------------------
    def get_floor(self, floor_id):
        return floor_id*4


    # ---------------------------------------------------------
    # distance_space
    # ---------------------------------------------------------
    # Euclidean distance between two points
    # ---------------------------------------------------------
    def distance_space(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (self.get_floor(p1[2]) - self.get_floor(p2[2]))**2)


    # ---------------------------------------------------------
    # get_location
    # ---------------------------------------------------------
    # Given a test fingerprint, return the estimated location (x,y,z) for this fingerprint
    # ---------------------------------------------------------
    def get_location(self, distances):
        # First: floor estimation
        floor_votes = np.zeros(self.n_floors)
        for i in range(self.k):
            best = np.argmin(distances)
            floor = int(self.train_locations[best, 2])
            floor_votes[floor] = floor_votes[floor] + 1
            distances[best] = 1000000  # big number

        estimated_floor = np.argmax(floor_votes)

        # Second: location estimation.
        # The distance is estimated only in train samples of the estimated floor
        index_floor = self.train_locations[:, 2] == estimated_floor
        distances_floor = distances[index_floor]

        x = 0
        y = 0
        for i in range(self.k):
            best = np.argmin(distances_floor)
            x = x + self.train_locations[best, 0]
            y = y + self.train_locations[best, 1]
            distances_floor[best] = 1000000  # big number

        x = x / self.k
        y = y / self.k

        return x, y, estimated_floor


    # ---------------------------------------------------------
    # get_locations
    # ---------------------------------------------------------
    # Get the location of a set of test_fingerprints
    # ---------------------------------------------------------
    def get_locations(self, test_fingerprints):
        distances = distance.cdist(self.train_fingerprints, test_fingerprints, "euclidean")
        n_test = test_fingerprints.shape[0]
        locations = np.zeros([n_test, 3])

        for i in range(n_test):
            x, y, z = self.get_location(distances[:, i])
            locations[i, 0] = x
            locations[i, 1] = y
            locations[i, 2] = z

        return locations


    # ---------------------------------------------------------
    # estimate_accuracy
    # ---------------------------------------------------------
    # Estimate the location accuracy of the estimated locations given the true ones.
    # ---------------------------------------------------------
    def estimate_accuracy(self, estimated_locations, true_locations):
        n_samples = estimated_locations.shape[0]
        verrors = np.zeros(n_samples)
        for i in range(n_samples):
            verrors[i] = self.distance_space(estimated_locations[i, :], true_locations[i, :])

        return verrors


    # ---------------------------------------------------------
    # get_accuracy
    # ---------------------------------------------------------
    # Get statistics of the accuracy of a set of test fingerprints
    # ---------------------------------------------------------
    def get_accuracy(self, test_fingerprints, test_locations):
        estimated_locations = self.get_locations(test_fingerprints)
        verrors = self.estimate_accuracy(estimated_locations, test_locations)

        return np.mean(verrors), np.percentile(verrors, 75)
