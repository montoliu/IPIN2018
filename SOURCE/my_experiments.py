# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

import my_functions as myf
import numpy as np
from sklearn.model_selection import KFold


# ----------------------------------------------
# This function is to test if indoorloc algoritm is working well
# We Test: month 01 vs month02, 01 vs 03, ..., 01 vs 15
# ----------------------------------------------
def exp_test_indoorloc(db_path):
    th = -80
    pct = 0.1
    k = 11

    fp_train, loc_train = myf.get_all_data_month(db_path, "01", 15, 0)
    v_index_train = myf.get_index_ap_with_data(fp_train, th, pct)

    for i in range(2, 16):
        month = "{0:02d}".format(i)
        fp_test, loc_test = myf.get_all_data_month(db_path, month, 0, 5)
        v_index_test = myf.get_index_ap_with_data(fp_test, th, pct)

        l_new, l_gone, l_both = myf.get_new_gone_both_aps(v_index_train, v_index_test)
        mean_acc, p75_acc = myf.go_common_approach(fp_train, fp_test, l_both, loc_train, loc_test, k)

        print(month + " " + str(len(l_both)) + " " + "{:0.2f}".format(mean_acc) + " " + "{:0.2f}".format(p75_acc), flush=True)


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
def exp_ap_gone(db_path):
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

    acc_common, p75_common = myf.go_common_approach(fp_month01, fp_month15, l_both, loc_month01, loc_month15, k)
    print("Using common approach (only in both sets " + str(l_both) + ") -> "
          + "{:0.2f}".format(acc_common)
          + " [" + "{:0.2f}".format(p75_common) + "] ")

    for l_gone in l_exps:
        l_all = list(l_both)
        l_all.extend(l_gone)

        acc_all, p75_all = myf.go_common_approach(fp_month01, fp_month15, l_all, loc_month01, loc_month15, k)
        acc_reg, p75_reg = myf.go_regression_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, method, k)
        acc_nei, p75_nei = myf.go_neighbours_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, k)
        acc_cmb, p75_cmb = myf.go_combination_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, k)

        print (str(l_gone) + " " + "{:0.2f}".format(acc_reg) + " "
                                    + "{:0.2f}".format(acc_nei) + " "
                                    + "{:0.2f}".format(acc_cmb) + " "
                                    + "{:0.2f}".format(acc_all) + " "
                                      + "{:0.2f}".format(p75_reg) + " "
                                      + "{:0.2f}".format(p75_nei) + " "
                                      + "{:0.2f}".format(p75_cmb) + " "
                                      + "{:0.2f}".format(p75_all), flush=True)


# ----------------------------------------------
# This experiment test which is the best AP combination of columns [16,50,51,70] to imputate one of [5,6,7,8,11]
# Only training data is used. 80% train, 20% test
# ----------------------------------------------
def exp_best_imputation_models_train(db_path):
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

                print(str(fold) + " " + str(l_comb) + " " + str(l_gone) + " "
                      + "{:0.2f}".format(acc_reg) + " "
                      + "{:0.2f}".format(acc_nei) + " "
                      + "{:0.2f}".format(p75_reg) + " "
                      + "{:0.2f}".format(p75_nei), flush=True)

                i = i + 1
            j = j + 1
        fold = fold + 1

    results_reg_50 = results_reg_50 / n_folds
    results_reg_75 = results_reg_75 / n_folds
    results_nei_50 = results_nei_50 / n_folds
    results_nei_75 = results_nei_75 / n_folds

    print("------")
    for i in range(len(l_exps)):
        for j in range(len(l_new)):
            print (str(l_exps[i]) + " " + str(l_new[j]) + " "
                   + "{:0.2f}".format(results_reg_50[i, j]) + " "
                   + "{:0.2f}".format(results_nei_50[i, j]) + " "
                   + "{:0.2f}".format(results_reg_75[i, j]) + " "
                   + "{:0.2f}".format(results_nei_75[i, j]))

