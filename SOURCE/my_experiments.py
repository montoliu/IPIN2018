# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

import my_functions as myf
import numpy as np
from sklearn.model_selection import KFold
import math

# ----------------------------------------------
# This function is to test if indoorloc algoritm is working well
# We test: month 01 vs month02, 01 vs 03, ..., 01 vs 15
# ----------------------------------------------
def exp_test_indoorloc(db_path, filename):
    th = -80
    pct = 0.1
    k = 11

    fp_train, loc_train = myf.get_all_data_month(db_path, "01", 15, 0)
    v_index_train = myf.get_index_ap_with_data(fp_train, th, pct)

    my_file = open(filename, 'wt')
    my_file.write("month, size_l_both, pct_50, pct_75\n")

    print("month, size_l_both, pct_50, pct_75")
    for i in range(2, 16):
        month = "{0:02d}".format(i)
        fp_test, loc_test = myf.get_all_data_month(db_path, month, 0, 5)
        v_index_test = myf.get_index_ap_with_data(fp_test, th, pct)

        l_new, l_gone, l_both = myf.get_new_gone_both_aps(v_index_train, v_index_test)
        mean_acc, p75_acc = myf.go_common_approach(fp_train, fp_test, l_both, loc_train, loc_test, k)

        s = month + "," + str(len(l_both)) + "," + "{:0.2f}".format(mean_acc) + "," + "{:0.2f}".format(p75_acc)
        print(s)
        my_file.write(s + "\n")

    my_file.close()


# ----------------------------------------------
# ----------------------------------------------
def exp_paper1(db_path):
    fp_month01, loc_month01 = myf.get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_month15 = myf.get_all_data_month(db_path, "15", 0, 5)

    method = "SVR"
    k = 7
    l_all = [5, 6, 7, 8, 11, 16, 21, 50, 51, 70]

    acc_ideal, p75_ideal = myf.go_common_approach(fp_month01, fp_month15, l_all, loc_month01, loc_month15, k)
    s = "{:0.2f}".format(acc_ideal) + "," + "{:0.2f}".format(p75_ideal)
    print(s)

    #print("gone,common_50,reg_50,nei_50,mean_50,max_50,zero_50,common_75,reg_75,nei_75,mean_75,max_75,zero_75")
    print("gone,common_50,reg_50,mean_50,common_75,reg_75,mean_75")

    acc_common = np.zeros((10, 1))
    acc_reg = np.zeros((10, 1))
    acc_nei = np.zeros((10, 1))
    acc_mean = np.zeros((10, 1))
    acc_max = np.zeros((10, 1))
    acc_zero = np.zeros((10, 1))
    p75_common = np.zeros((10, 1))
    p75_reg = np.zeros((10, 1))
    p75_nei = np.zeros((10, 1))
    p75_mean = np.zeros((10, 1))
    p75_max = np.zeros((10, 1))
    p75_zero = np.zeros((10, 1))

    for i in range(len(l_all)):
        l_both = []
        l_gone = [l_all[i]]
        for j in range(len(l_all)):
            if i != j:
                l_both.append(l_all[j])

        train_fps_norm, test_fps_norm = myf.get_normalized_data(fp_month01, fp_month15, l_both, l_gone)
        size_l_gone = len(l_gone)
        size_l_both = len(l_both)

        acc_common[i, 0], p75_common[i, 0] = myf.go_common_approach(fp_month01, fp_month15, l_both, loc_month01, loc_month15, k)
        acc_reg[i, 0], p75_reg[i, 0] = myf.go_regression_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, method, k)
        #acc_nei[i, 0], p75_nei[i, 0] = myf.go_neighbours_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, k)
        acc_mean[i, 0], p75_mean[i, 0] = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "mean", k)
        #acc_max[i, 0], p75_max[i, 0] = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "max", k)
        #acc_zero[i, 0], p75_zero[i, 0] = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "zero", k)

        #s1 = str(l_gone) + ","
        #s2 = "{:0.2f}".format(acc_common[i, 0]) + "," + "{:0.2f}".format(acc_reg[i, 0]) + "," + "{:0.2f}".format(acc_nei[i, 0]) + ","
        #s3 = "{:0.2f}".format(acc_mean[i, 0]) + "," + "{:0.2f}".format(acc_max[i, 0]) + "," + "{:0.2f}".format(acc_zero[i, 0]) + ","
        #s4 = "{:0.2f}".format(p75_common[i, 0]) + "," + "{:0.2f}".format(p75_reg[i, 0]) + "," + "{:0.2f}".format(p75_nei[i, 0]) + ","
        #s5 = "{:0.2f}".format(p75_mean[i, 0]) + "," + "{:0.2f}".format(p75_max[i, 0]) + "," + "{:0.2f}".format(p75_zero[i, 0])
        #print(s1 + s2 + s3 + s4 + s5)

        s1 = str(l_gone) + ","
        s2 = "{:0.2f}".format(acc_common[i, 0]) + "," + "{:0.2f}".format(acc_reg[i, 0]) + "," + "{:0.2f}".format(acc_mean[i, 0]) + ","
        s3 = "{:0.2f}".format(p75_common[i, 0]) + "," + "{:0.2f}".format(p75_reg[i, 0]) + "," + "{:0.2f}".format(p75_mean[i, 0])
        print(s1 + s2 + s3)

    #s2 = "{:0.2f}".format(np.mean(acc_common)) + "," + "{:0.2f}".format(np.mean(acc_reg)) + "," + "{:0.2f}".format(np.mean(acc_nei)) + ","
    #s3 = "{:0.2f}".format(np.mean(acc_mean)) + "," + "{:0.2f}".format(np.mean(acc_max)) + "," + "{:0.2f}".format(np.mean(acc_zero)) + ","
    #s4 = "{:0.2f}".format(np.mean(p75_common)) + "," + "{:0.2f}".format(np.mean(p75_reg)) + "," + "{:0.2f}".format(np.mean(p75_nei)) + ","
    #s5 = "{:0.2f}".format(np.mean(p75_mean)) + "," + "{:0.2f}".format(np.mean(p75_max)) + "," + "{:0.2f}".format(np.mean(p75_zero))
    #print(s2 + s3 + s4 + s5)
    s1 = "{:0.2f}".format(np.mean(acc_common)) + "," + "{:0.2f}".format(np.mean(acc_reg)) + "," + "{:0.2f}".format(np.mean(acc_mean)) + ","
    s2 = "{:0.2f}".format(np.mean(p75_common)) + "," + "{:0.2f}".format(np.mean(p75_reg)) + "," + "{:0.2f}".format(np.mean(p75_mean))
    print(s1 + s2)


# ----------------------------------------------
# ----------------------------------------------
def exp_paper2(db_path, k):
    fp_month01, loc_train = myf.get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_test = myf.get_all_data_month(db_path, "15", 0, 5)
    fp_month01_norm, fp_month15_norm = myf.normalize_data(fp_month01, fp_month15)

    l_both = [16, 50, 51, 70]
    l_gone = [5, 6, 7, 8, 11, 21]
    l_all = list(l_both)
    l_all.extend(l_gone)

    common_train_data = fp_month01_norm[:,l_both]
    common_test_data = fp_month15_norm[:, l_both]

    ideal_train_data = fp_month01_norm[:, l_all]
    ideal_test_data = fp_month15_norm[:, l_all]

    acc_common, p75_common = myf.ips(common_train_data, common_test_data, loc_train, loc_test, k)
    acc_ideal, p75_ideal = myf.ips(ideal_train_data, ideal_test_data, loc_train, loc_test, k)

    new_test_data_reg = np.zeros((fp_month15_norm.shape[0], len(l_gone)))
    new_test_data_mean = np.zeros((fp_month15_norm.shape[0], len(l_gone)))

    i = 0
    for wap in l_gone:
        # reg
        x_train = common_train_data
        y_train = fp_month01_norm[:, wap]
        x_test = fp_month15_norm[:, l_both]
        new_test_data_reg[:, i] = myf.impute_regression(x_train, y_train, x_test)

        #mean
        mean_wap = np.mean(y_train)
        new_test_data_mean[:, i] = mean_wap
        i = i + 1

    reg_train_data = ideal_train_data
    reg_test_data = np.concatenate((fp_month15_norm[:, l_both], new_test_data_reg), axis=1)
    acc_reg, p75_reg = myf.ips(reg_train_data, reg_test_data, loc_train, loc_test, k)

    mean_train_data = ideal_train_data
    mean_test_data = np.concatenate((fp_month15_norm[:, l_both], new_test_data_mean), axis=1)
    acc_mean, p75_mean = myf.ips(mean_train_data, mean_test_data, loc_train, loc_test, k)

    s1 = "{:0.2f}".format(acc_ideal) + "," + "{:0.2f}".format(acc_common) + "," + "{:0.2f}".format(acc_reg) + "," + "{:0.2f}".format(acc_mean)
    s2 = "{:0.2f}".format(p75_ideal) + "," + "{:0.2f}".format(p75_common) + "," + "{:0.2f}".format(p75_reg) + "," + "{:0.2f}".format(p75_mean)
    s = s1 + "," + s2
    print(s)


# ----------------------------------------------
# ----------------------------------------------
def exp_paper3(db_path, k):
    fp_month01, loc_train = myf.get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_test = myf.get_all_data_month(db_path, "15", 0, 5)
    fp_month01_norm, fp_month15_norm = myf.normalize_data(fp_month01, fp_month15)

    l_both = [16, 50, 51, 70]
    #l_gone = [5, 6, 7, 8, 11, 21]
    l_gone = [21, 11, 8, 7, 6, 5]
    l_all = list(l_both)
    l_all.extend(l_gone)

    common_train_data = fp_month01_norm[:,l_both]
    common_test_data = fp_month15_norm[:, l_both]

    ideal_train_data = fp_month01_norm[:, l_all]
    ideal_test_data = fp_month15_norm[:, l_all]

    acc_common, p75_common = myf.ips(common_train_data, common_test_data, loc_train, loc_test, k)
    acc_ideal, p75_ideal = myf.ips(ideal_train_data, ideal_test_data, loc_train, loc_test, k)

    new_test_data_reg = np.zeros((fp_month15_norm.shape[0], len(l_gone)))
    new_test_data_mean = np.zeros((fp_month15_norm.shape[0], len(l_gone)))

    i = 0
    x_train = common_train_data
    x_test = fp_month15_norm[:, l_both]

    for wap in l_gone:
        y_train = fp_month01_norm[:, wap]
        y_test = myf.impute_regression(x_train, y_train, x_test)
        new_test_data_reg[:, i] = y_test

        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        mean_wap = np.mean(y_train)
        new_test_data_mean[:, i] = mean_wap

        x_train = np.concatenate((x_train, y_train), axis=1)
        x_test = np.concatenate((x_test, y_test), axis=1)
        i = i + 1

    reg_train_data = ideal_train_data
    reg_test_data = np.concatenate((fp_month15_norm[:, l_both], new_test_data_reg), axis=1)
    acc_reg, p75_reg = myf.ips(reg_train_data, reg_test_data, loc_train, loc_test, k)

    mean_train_data = ideal_train_data
    mean_test_data = np.concatenate((fp_month15_norm[:, l_both], new_test_data_mean), axis=1)
    acc_mean, p75_mean = myf.ips(mean_train_data, mean_test_data, loc_train, loc_test, k)

    s1 = "{:0.2f}".format(acc_ideal) + "," + "{:0.2f}".format(acc_common) + "," + "{:0.2f}".format(acc_reg) + "," + "{:0.2f}".format(acc_mean)
    s2 = "{:0.2f}".format(p75_ideal) + "," + "{:0.2f}".format(p75_common) + "," + "{:0.2f}".format(p75_reg) + "," + "{:0.2f}".format(p75_mean)
    s = s1 + "," + s2
    print(s)


# ----------------------------------------------
# ----------------------------------------------
def exp_paper4(db_path, k, pct):
    fp_month01, loc_train = myf.get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_test = myf.get_all_data_month(db_path, "15", 0, 5)
    fp_month01_norm, fp_month15_norm = myf.normalize_data(fp_month01, fp_month15)

    n_test_samples_for_modeling = math.floor(fp_month15_norm.shape[0] * pct)
    fp_test_pre = fp_month15_norm[0:n_test_samples_for_modeling, :]
    fp_test_post = fp_month15_norm[n_test_samples_for_modeling:fp_month15_norm.shape[0], :]

    l_both = [16, 50, 51, 70]
    l_gone = [5, 6, 7, 8, 11, 21]
    l_all = list(l_both)
    l_all.extend(l_gone)

    common_train_data = fp_month01_norm[:, l_both]
    common_test_data = fp_test_post[:, l_both]

    ideal_train_data = fp_month01_norm[:, l_all]
    ideal_test_data = fp_test_post[:, l_all]

    acc_common, p75_common = myf.ips(common_train_data, common_test_data, loc_train, loc_test, k)
    acc_ideal, p75_ideal = myf.ips(ideal_train_data, ideal_test_data, loc_train, loc_test, k)

    acc_reg = 0
    p75_reg = 0
    acc_mean = 0
    p75_mean = 0
    s1 = "{:0.2f}".format(acc_ideal) + "," + "{:0.2f}".format(acc_common) + "," + "{:0.2f}".format(acc_reg) + "," + "{:0.2f}".format(acc_mean)
    s2 = "{:0.2f}".format(p75_ideal) + "," + "{:0.2f}".format(p75_common) + "," + "{:0.2f}".format(p75_reg) + "," + "{:0.2f}".format(p75_mean)
    s = s1 + "," + s2
    print(s)

# ----------------------------------------------
# Train: month 01 (train files)
# Test: month 15 (test files)
# CASE 1:
# 01 columns: 8, 16, 50, 51, 71
# 15 columns: gone 8
# CASE 2:
# 01 columns: 5, 16, 50, 51, 71
# 15 columns: gone 5
# CASE 3:
# 01 columns: 11, 16, 50, 51, 71
# 15 columns: gone 11
# ----------------------------------------------
def exp_ap_gone(db_path, filename):
    fp_month01, loc_month01 = myf.get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_month15 = myf.get_all_data_month(db_path, "15", 0, 5)

    method = "SVR"
    k = 11

    l_both = [16, 50, 51, 70]
    l_exps = [[5], [6], [7], [8], [11],
              [5, 6], [5, 7], [5, 8], [5, 11], [6, 7], [6, 8], [6, 11], [7, 8], [7, 11], [8, 11],
              [5, 6, 7], [5, 6, 8], [5, 6, 11], [5, 7, 8], [5, 7, 11], [5, 8, 11],
              [5, 6, 7, 8], [5, 6, 7, 11],
              [5, 6, 7, 8, 11]]

    my_file = open(filename, 'wt')
    print("gone,all_50,reg_50,nei_50,cmb_50,mean_50,max_50,zero_50,all_75,reg_75,nei_75,cmb_75,mean_75,max_75,zero_75")
    my_file.write("gone,all_50,reg_50,nei_50,cmb_50,mean_50,max_50,zero_50,all_75,reg_75,nei_75,cmb_75,mean_75,max_75,zero_75\n")

    acc_common, p75_common = myf.go_common_approach(fp_month01, fp_month15, l_both, loc_month01, loc_month15, k)
    s = "Using common approach (only in both sets " + str(l_both) + ") -> " + "{:0.2f}".format(acc_common) + " [" + "{:0.2f}".format(p75_common) + "] "
    print(s)
    my_file.write(s + "\n")

    for l_gone in l_exps:
        l_all = list(l_both)
        l_all.extend(l_gone)

        train_fps_norm,  test_fps_norm = myf.get_normalized_data(fp_month01, fp_month15, l_both, l_gone)
        size_l_gone = len(l_gone)
        size_l_both = len(l_both)

        acc_all, p75_all = myf.go_common_approach(fp_month01, fp_month15, l_all, loc_month01, loc_month15, k)
        acc_mean, p75_mean = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "mean", k)
        acc_max, p75_max = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "max", k)
        acc_zero, p75_zero = myf.go_basic_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, "zero", k)
        acc_reg, p75_reg = myf.go_regression_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, method, k)
        acc_nei, p75_nei = myf.go_neighbours_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, k)
        acc_cmb, p75_cmb = myf.go_combination_approach(train_fps_norm, test_fps_norm, loc_month01, loc_month15, size_l_gone, size_l_both, k)

        s1 = str(l_gone) + ","
        s2 = "{:0.2f}".format(acc_all) + "," + "{:0.2f}".format(acc_reg) + "," + "{:0.2f}".format(acc_nei) + "," + "{:0.2f}".format(acc_cmb) + ","
        s3 = "{:0.2f}".format(acc_mean) + "," + "{:0.2f}".format(acc_max) + "," + "{:0.2f}".format(acc_zero) + ","
        s4 = "{:0.2f}".format(p75_all) + "," + "{:0.2f}".format(p75_reg) + "," + "{:0.2f}".format(p75_nei) + "," + "{:0.2f}".format(p75_cmb) + ","
        s5 = "{:0.2f}".format(p75_mean) + "," + "{:0.2f}".format(p75_max) + "," + "{:0.2f}".format(p75_zero)
        print(s1 + s2 + s3 + s4 + s5)
        my_file.write(s1 + s2 + s3 + s4 + s5 + "\n")

    my_file.close()


# ----------------------------------------------
# This experiment test which is the best AP combination of columns [16,50,51,70] to imputate one of [5,6,7,8,11]
# Only training data is used. 80% train, 20% test
# ----------------------------------------------
def exp_best_imputation_models_train(db_path, filename):

    method = "SVR"
    n_folds = 5
    k = 11

    l_exps = [[16], [50], [51], [70], [16, 50], [16, 51], [16, 70], [50, 51], [50, 70], [51, 70],
              [16, 50, 51], [16, 50, 70], [16, 51, 70], [50, 51, 70], [16, 50, 51, 70]]
    l_new = [[5], [6], [7], [8], [11]]

    fp_all_train, loc_all_train = myf.get_all_data_month(db_path, "01", 15, 0)

    results_reg_50 = np.zeros((len(l_exps),len(l_new)))
    results_reg_75 = np.zeros((len(l_exps),len(l_new)))
    results_nei_50 = np.zeros((len(l_exps),len(l_new)))
    results_nei_75 = np.zeros((len(l_exps),len(l_new)))

    fold = 1

    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(fp_all_train):
        fp_train, fp_test = fp_all_train[train_index], fp_all_train[test_index]
        loc_train, loc_test = loc_all_train[train_index], loc_all_train[test_index]

        j = 0
        for l_gone in l_new:
            i = 0
            for l_comb in l_exps:
                acc_reg, p75_reg = myf.go_regression_approach(fp_train, fp_test, loc_train, loc_test, l_gone, l_comb, method, k)
                acc_nei, p75_nei = myf.go_neighbours_approach(fp_train, fp_test, loc_train, loc_test, l_gone, l_comb, k)

                results_reg_50[i, j] += acc_reg
                results_reg_75[i, j] += p75_reg
                results_nei_50[i, j] += acc_nei
                results_nei_75[i, j] += p75_nei

                s = str(fold) + "," + str(l_comb) + "," + str(l_gone) + "," + "{:0.2f}".format(acc_reg) + "," + "{:0.2f}".format(acc_nei) + "," + "{:0.2f}".format(p75_reg) + "," + "{:0.2f}".format(p75_nei)
                print(s)

                i = i + 1
            j = j + 1
        fold = fold + 1

    results_reg_50 = results_reg_50 / n_folds
    results_reg_75 = results_reg_75 / n_folds
    results_nei_50 = results_nei_50 / n_folds
    results_nei_75 = results_nei_75 / n_folds

    print("------")
    my_file = open(filename, 'wt')
    print("X Y reg_50 nei_50 reg_75 nei_75")
    my_file.write("X Y reg_50 nei_50 reg_75 nei_75\n")

    for j in range(len(l_new)):
        for i in range(len(l_exps)):
            s = str(l_exps[i]) + "," + str(l_new[j]) + "," + "{:0.2f}".format(results_reg_50[i, j]) + "," + "{:0.2f}".format(results_nei_50[i, j]) + "," + "{:0.2f}".format(results_reg_75[i, j]) + "," + "{:0.2f}".format(results_nei_75[i, j])
            print(s)
            my_file.write(s)

    my_file.close()
