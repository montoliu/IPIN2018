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
    # ---------------------------------------------------------
    def __init__(self, train_fingerprints, train_locations, k=3, penalty_floor=4):
        self.train_fingerprints = train_fingerprints
        self.train_locations = train_locations
        self.n_training_samples = train_fingerprints.shape[0]
        self.n_aps = train_fingerprints.shape[1]
        self.n_floors = int(np.max(self.train_locations[:,2]) + 1)
        self.k = k
        self.penalty_floor = penalty_floor


    # ---------------------------------------------------------
    # distance_fingerprint
    # ---------------------------------------------------------
    # Return the euclidean distance between two fingerprints fp1, fp2
    # fp1 and pf2 are np arrays
    # ---------------------------------------------------------
    def distance_fingerprint(self, fp1, fp2):
        n = fp1.shape[0]
        d = 0

        for i in range(n):
            d = d + abs(fp1[i] - fp2[i])

        return d / n


    # ---------------------------------------------------------
    # distance_space
    # ---------------------------------------------------------
    def distance_space(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    # ---------------------------------------------------------
    # get_location
    # ---------------------------------------------------------
    # Given a test fingerprint, return the estimated location (x,y,z) for this fingerprint
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
        # The distance is estimated only in train samples of this floor
        index_floor = self.train_locations[:, 2] == estimated_floor
        distances_floor = distances[index_floor]

        x = 0
        y = 0
        for i in range(self.k):
            best = np.argmin(distances_floor)
            x = x + self.train_locations[best][0]
            y = y + self.train_locations[best][1]
            distances_floor[best] = 1000000  # big number

        x = x / self.k
        y = y / self.k

        return x, y, estimated_floor


    # ---------------------------------------------------------
    # get_locations
    # ---------------------------------------------------------
    def get_locations(self, test_fingerprints):
        distances = distance.cdist(self.train_fingerprints, test_fingerprints, "euclidean")
        n_test = test_fingerprints.shape[0]
        locations = np.zeros([n_test,3])

        for i in range(n_test):
            x,y,z = self.get_location(distances[:,i])
            locations[i][0] = x
            locations[i][1] = y
            locations[i][2] = z

        return locations


    # ---------------------------------------------------------
    # estimate_accuracy
    # ---------------------------------------------------------
    # Estimate the location accuracy of the estimated locations given the true ones.
    # If the floor has not been estimated a penality of 4 meters is added.
    def estimate_accuracy(self, estimated_locations, true_locations):
        n_samples = estimated_locations.shape[0]
        verrors = np.zeros(n_samples)
        for i in range(n_samples):
            verrors[i] = self.distance_space(estimated_locations[i][0], estimated_locations[i][1], true_locations[i][0], true_locations[i][1])
            #if estimated_locations[i][2] != true_locations[i][2]:
            #    verrors[i] = verrors[i] + self.penalty_floor

        return np.sum(verrors) / n_samples, verrors


    # ---------------------------------------------------------
    # get_accuracy
    # ---------------------------------------------------------
    def get_accuracy(self, test_fingerprints, test_locations):
        estimated_locations = self.get_locations(test_fingerprints)
        accuracy, verrors = self.estimate_accuracy(estimated_locations,test_locations)

        return accuracy, verrors
