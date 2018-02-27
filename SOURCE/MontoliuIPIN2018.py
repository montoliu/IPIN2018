# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

import my_experiments as mye
import sys


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    db_path = "../DB/db/"

    exp = 3

    if exp == 1:
        filename = "exp_test_indoorloc.txt"
        sys.stdout = open(filename, 'wt')
        print("month size_l_both accuracy pct_75")
        mye.exp_test_indoorloc(db_path)
    elif exp == 2:
        filename = "exp_ap_gone.txt"
        sys.stdout = open(filename, 'wt')
        print("gone reg_50 nei_50 cmb_50 all_50 reg_75 nei_75 cmb_75 all_75")
        mye.exp_ap_gone(db_path)
    elif exp == 3:
        filename = "exp_best_imputation_models_train.txt"
        sys.stdout = open(filename, 'wt')
        print("X Y reg_50 nei_50 reg_75 nei_75")
        mye.exp_best_imputation_models_train(db_path)









