# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

import numpy as np
import pandas as pd

from scipy.spatial import distance

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors

import matplotlib.pyplot as plt
import indoorLoc as IPS


# -----------------------------------------------------------
# cdf plot
# -----------------------------------------------------------
def cdf(data):
    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf,linestyle='--', marker="o", color='b')
    plt.ylim((0,1))
    plt.ylabel("CDF")
    plt.grid(True)

    plt.show()

# -----------------------------------------------------------
# get_stats_aps
# -----------------------------------------------------------
# Return a np array of dim n_aps x 3
# [0] -> pct of samples with data
# [1] -> pct of samples with rssi > TH1  (strong signal, i.e. -50)
def get_stats_aps(fps, th1):
    n_samples = fps.shape[0]   # number of rows == number of samples
    n_aps = fps.shape[1]       # number of columns == number of APS
    stats_aps = np.zeros([n_aps,3])

    idx_no_100 = fps != 100
    idx_th1 = np.logical_and(fps != 100, fps > th1)

    stats_aps[:,0] = sum(idx_no_100)/n_samples
    stats_aps[:,1] = sum(idx_th1)/n_samples

    return stats_aps


# -----------------------------------------------------------
# get_index_ap_with_data
# -----------------------------------------------------------
# Return an array of n_aps dimension
# The ith element is 0 if all sample of the ith AP (column) is 100 (no RSSI data received from this AP)
# or there are less than pct samples with rssi value < th
# 1 otherwise, i.e. are reliable APs
# -----------------------------------------------------------
def get_index_ap_with_data(fps, th, pct):
    stats_aps = get_stats_aps(fps, th)
    strong_aps = stats_aps[:,1] >= pct
    v_index = np.zeros(fps.shape[1])
    v_index[strong_aps] = 1

    return v_index


# -----------------------------------------------------------
# get_new_gone_aps
# -----------------------------------------------------------
def get_new_gone_both_aps(v_index_db1, v_index_db2):
    l_new = []
    l_gone = []
    l_both = []
    n_aps = v_index_db1.shape[0]
    for i in range(n_aps):
        if v_index_db1[i] == 1 and v_index_db2[i] == 0:
            l_gone.append(i)
        elif v_index_db1[i] == 0 and v_index_db2[i] == 1:
            l_new.append(i)
        elif v_index_db1[i] == 1 and v_index_db2[i] == 1:
            l_both.append(i)

    return l_new, l_gone, l_both


# -----------------------------------------------------------
# get_data
# -----------------------------------------------------------
def get_data(db_path, str_month, n_files, type):
    # read first file to initialize the data variables
    fp_filename = db_path + str_month + "/" + type + "01rss.csv"
    loc_filename = db_path + str_month + "/" + type + "01crd.csv"

    ifp = pd.read_csv(fp_filename, header=None)
    iloc = pd.read_csv(loc_filename, header=None)

    fp_data = ifp.values
    loc_data = iloc.values

    # Read the rest of  files
    for i in range(2, n_files + 1):
        fp_filename = db_path + str_month + "/" + type + "{0:02d}".format(i) + "rss.csv"
        loc_filename = db_path + str_month + "/" + type + "{0:02d}".format(i) + "crd.csv"

        ifp = pd.read_csv(fp_filename, header=None)
        iloc = pd.read_csv(loc_filename, header=None)

        fp_data = np.concatenate((fp_data, ifp.values))
        loc_data = np.concatenate((loc_data, iloc.values))

    return fp_data, loc_data


# -----------------------------------------------------------
# get_all_data_month
# -----------------------------------------------------------
# This function create a numpy array with all the samples of the data from one month.
# It concatenates the samples of all training and test files.
# Return two numpy arrays: fingerprints and locations
# NOTE: if n_test_files == 0 then no test files are used.
# NOTE: if n_train_files == 0 then no train files are used.
def get_all_data_month(db_path, str_month, n_train_files, n_test_files):

    if n_train_files > 0:
        fp_data_train, loc_data_train = get_data(db_path, str_month, n_train_files, "trn")

    if n_test_files > 0:
        fp_data_test, loc_data_test = get_data(db_path, str_month, n_test_files, "tst")

    if n_train_files > 0 and n_test_files > 0:
        fp_data = np.concatenate((fp_data_train, fp_data_test))
        loc_data = np.concatenate((loc_data_train, loc_data_test))
    elif n_train_files ==0 and n_test_files > 0:
        fp_data = fp_data_test
        loc_data = loc_data_test
    elif n_train_files > 0 and n_test_files == 0:
        fp_data = fp_data_train
        loc_data = loc_data_train

    # Since floor id are 3 and 5, but they are consectutive in the building, we replace the 3 by 0, and the 5 by 1
    idx_floor_3 = loc_data[:, 2] == 3
    idx_floor_5 = loc_data[:, 2] == 5

    loc_data[idx_floor_3, 2] = 0
    loc_data[idx_floor_5, 2] = 1

    return fp_data, loc_data


# -----------------------------------------------------------
# normalize01
# -----------------------------------------------------------
# 100 -> 0
# -x -> between 0 and 1
# 0 means no data or very poor signal.
# 1 means very strong signal
def normalize01(data, min_value):
    index_100 = data == 100
    new_data = (data + abs(min_value)) / abs(min_value)
    new_data[index_100] = 0
    return new_data


# -----------------------------------------------------------
# go_common_approach
# -----------------------------------------------------------
# It uses APs that exists in both sets
def go_common_approach(db1, db2, l_both, train_locations, test_locations, k=3):
    train_fingerprints = db1[:, l_both]
    test_fingerprints = db2[:, l_both]
    min_rssi = np.amin(train_fingerprints)

    train_fingerprints_norm = normalize01(train_fingerprints, min_rssi)
    test_fingerprints_norm = normalize01(test_fingerprints, min_rssi)

    indoorloc_model = IPS.IndoorLoc(train_fingerprints_norm, train_locations, k)
    mean_acc, p75_acc = indoorloc_model.get_accuracy(test_fingerprints_norm, test_locations)

    return mean_acc, p75_acc


# -----------------------------------------------------------
# get_new_aps_by_regression
# -----------------------------------------------------------
def get_new_aps_by_regression(train_fps_norm, test_fps_norm, l_both, l_gone_aps, method):
    test_y = np.zeros([test_fps_norm.shape[0], len(l_gone_aps)])
    for i in range(len(l_gone_aps)):
        train_x = train_fps_norm[:, 0:len(l_both)]
        train_y = train_fps_norm[:, len(l_both) + i]

        # generate regression model and predict test new column
        if method == "LR":
            lin_reg_model = linear_model.LinearRegression()
            lin_reg_model.fit(train_x, train_y)
            data = lin_reg_model.predict(test_fps_norm)
        elif method == "SVR":
            svr_rbf = SVR(kernel='rbf', C=1, gamma=0.15)
            model = svr_rbf.fit(train_x, train_y)
            data = model.predict(test_fps_norm)
        elif method == "MLP":
            mlp = MLPRegressor()
            model = mlp.fit(train_x, train_y)
            data = model.predict(test_fps_norm)
        elif method == "RF":
            rf = RandomForestRegressor()
            model = rf.fit(train_x, train_y)
            data = model.predict(test_fps_norm)
        elif method == "3NN":
            knn = neighbors.KNeighborsRegressor(3)
            model = knn.fit(train_x, train_y)
            data = model.predict(test_fps_norm)
        elif method == "5NN":
            knn = neighbors.KNeighborsRegressor(5)
            model = knn.fit(train_x, train_y)
            data = model.predict(test_fps_norm)

        test_y[:, i] = data

    return test_y


# -----------------------------------------------------------
# get_new_aps_by_neighbours
# -----------------------------------------------------------
def get_new_aps_by_neighbours(train_fps_norm, test_fps_norm, l_all, l_both, l_gone_aps, k):
    train_x = train_fps_norm[:, 0:len(l_both)]
    train_y = train_fps_norm[:, len(l_both):len(l_all)]
    test_x = test_fps_norm[:, 0:len(l_both)]
    test_y = np.zeros([test_x.shape[0],len(l_gone_aps)])

    distances = distance.cdist(train_x, test_x, "euclidean")
    for i in range(test_x.shape[0]):

        for ik in range(k):
            best = np.argmin(distances[:, i])
            test_y[i,:] = test_y[i, :] + train_y[best, :]
            distances[best,i] = 1000000 #big number

        test_y[i, :] = test_y[i, :] / k

    return test_y


# -----------------------------------------------------------
# go_regression_approach
# -----------------------------------------------------------
# This approach uses a regression technique to estimate the values of the gone AP in test.
def go_regression_approach(db1, db2, train_locations, test_locations, l_gone_aps, l_both, method, k=3):
    l_all = list(l_both)
    l_all.extend(l_gone_aps)
    train_fps = db1[:, l_all]
    test_fps = db2[:, l_both]

    min_rssi = np.amin(train_fps)

    train_fps_norm = normalize01(train_fps, min_rssi)
    test_fps_norm = normalize01(test_fps, min_rssi)

    new_data_aps = get_new_aps_by_regression(train_fps_norm, test_fps_norm, l_both, l_gone_aps, method)
    new_test_fps_norm = np.concatenate((test_fps_norm, new_data_aps), axis=1)

    indoorloc_model = IPS.IndoorLoc(train_fps_norm, train_locations, k)
    mean_acc, p75_acc = indoorloc_model.get_accuracy(new_test_fps_norm, test_locations)

    return mean_acc, p75_acc


# -----------------------------------------------------------
# go_neighbour_approach
# -----------------------------------------------------------
# This approach uses a technique based on nearest neighbours to estimate the values of the gone AP in test.
# For each test sample, asign to the columns values the ones in the k-th most similar in the train set
def go_neighbours_approach(db1, db2, train_locations, test_locations, l_gone_aps, l_both, k):
    l_all = list(l_both)
    l_all.extend(l_gone_aps)
    train_fps = db1[:, l_all]
    test_fps = db2[:, l_both]

    min_rssi = np.amin(train_fps)

    train_fps_norm = normalize01(train_fps, min_rssi)
    test_fps_norm = normalize01(test_fps, min_rssi)

    new_data_aps = get_new_aps_by_neighbours(train_fps_norm, test_fps_norm, l_all, l_both, l_gone_aps, k)
    new_test_fps_norm = np.concatenate((test_fps_norm, new_data_aps), axis=1)

    indoorloc_model = IPS.IndoorLoc(train_fps_norm, train_locations, k)
    mean_acc, p75_acc = indoorloc_model.get_accuracy(new_test_fps_norm, test_locations)

    return mean_acc, p75_acc


# -----------------------------------------------------------
# go_combination_approach
# -----------------------------------------------------------
# Combine go_regression_approach (with SVR) with go_neighbours_approach
# The nex value is the mean of the two estimated using the previous methods
def go_combination_approach(db1, db2, train_locations, test_locations, l_gone_aps, l_both, k):
    l_all = list(l_both)
    l_all.extend(l_gone_aps)
    train_fps = db1[:, l_all]
    test_fps = db2[:, l_both]

    min_rssi = np.amin(train_fps)

    train_fps_norm = normalize01(train_fps, min_rssi)
    test_fps_norm = normalize01(test_fps, min_rssi)

    new_data_aps_reg = get_new_aps_by_regression(train_fps_norm, test_fps_norm, l_both, l_gone_aps, "SVR")
    new_data_aps_nei = get_new_aps_by_neighbours(train_fps_norm, test_fps_norm, l_all, l_both, l_gone_aps, k)

    new_data_aps = (new_data_aps_reg + new_data_aps_nei) / 2

    new_test_fps_norm = np.concatenate((test_fps_norm, new_data_aps), axis=1)

    indoorloc_model = IPS.IndoorLoc(train_fps_norm, train_locations, k)
    mean_acc, p75_acc= indoorloc_model.get_accuracy(new_test_fps_norm, test_locations)

    return mean_acc, p75_acc






