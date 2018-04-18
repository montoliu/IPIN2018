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

    exp = 7
    k = 1
    pct = 0.2

    if exp == 1:
        output_filename = "exp_test_indoorloc.txt"
        mye.exp_test_indoorloc(db_path, output_filename)
    elif exp == 2:
        output_filename = "exp_ap_gone.txt"
        mye.exp_ap_gone(db_path, output_filename)
    elif exp == 3:
        output_filename = "exp_best_imputation_models_train.txt"
        mye.exp_best_imputation_models_train(db_path, output_filename)
    elif exp == 4:
        mye.exp_paper1(db_path)
    elif exp == 5:
        mye.exp_paper2(db_path, k)
    elif exp == 6:
        mye.exp_paper3(db_path, k)
    elif exp == 7:
        mye.exp_paper4(db_path, k, pct)







