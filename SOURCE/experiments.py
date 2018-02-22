# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

from functions import get_all_data_month
from functions import go_common_approach
from functions import go_regression_approach
from functions import go_neighbours_approach
from functions import go_combination_approach


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
    fp_month01, loc_month01 = get_all_data_month(db_path, "01", 15, 0)
    fp_month15, loc_month15 = get_all_data_month(db_path, "15", 0, 5)

    method = "SVR"
    k = 11

    l_both = [16, 50, 51, 70]
    l_exps = [[5], [6], [7], [8], [11], [5,6], [5,7], [5,8], [5,11], [5,6,7], [5,6,8], [5,6,7,8,11]]

    acc_common = go_common_approach(fp_month01, fp_month15, l_both, loc_month01, loc_month15)
    print("Using common approach (only in both sets " + str(l_both) + ") -> " + str(acc_common))
    print("------------------")

    for l_gone in l_exps:
        l_all = list(l_both)
        l_all.extend(l_gone)

        acc_all = 1 #go_common_approach(fp_month01, fp_month15, l_all, loc_month01, loc_month15)
        acc_reg = 1 #go_regression_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, method)
        acc_nei = 1#go_neighbours_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, k)
        acc_cmb = go_combination_approach(fp_month01, fp_month15, loc_month01, loc_month15, l_gone, l_both, k)

        print (str(l_gone) + " -> " + "{:0.2f}".format(acc_reg) + " "
                                    + "{:0.2f}".format(acc_nei) + " "
                                    + "{:0.2f}".format(acc_cmb) + " "
                                    + "{:0.2f}".format(acc_all))

