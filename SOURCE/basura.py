# This code is the one used in the paper:
# R. Montoliu et al. "A new methodology for long term maintenance of WiFi fingerprinting radio maps"
# Proceedings of the 9th International Conference on Indoor Positioning, IPIN 2018.

import numpy as np
import pandas as pd

from scipy.spatial import distance
from my_functions import get_all_data_month

# own code
from indoorLoc import IndoorLoc
from my_experiments import exp_ap_gone
from my_experiments import exp_test_indoorloc






# -----------------------------------------------------------
# filter_aps_by_th
# -----------------------------------------------------------
# This function returns a new dataset with the same number of samples
# but with only the APs (columns) with at least a pct samples with rssi > th
def filter_aps_by_th(fps, th, pct):
    stats_aps = get_stats_aps(fps, th)

    good_aps = stats_aps[:,1] >= pct

    return fps[:,good_aps]











# -----------------------------------------------------------
# The objective of this experiment is to obtain some data the importance of each AP in each month
# Then it will possible to determine the useful APS and the ones that are noise
# -----------------------------------------------------------
def Experiment1(db_path):

    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 5)
    stats_aps = get_stats_aps(fingerprints_month1,-80)

    df = pd.DataFrame(stats_aps)
    df.to_csv("ap01.csv")

    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "02", 1, 5)
    stats_aps = get_stats_aps(fingerprints_month1,-80)

    df = pd.DataFrame(stats_aps)
    df.to_csv("ap02.csv")


    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "15", 1, 5)
    stats_aps = get_stats_aps(fingerprints_month1,-80)

    df = pd.DataFrame(stats_aps)
    df.to_csv("ap15.csv")


# -----------------------------------------------------------
# How many have in common after filtering by th
# -----------------------------------------------------------
def Experiment4(db_path):

    fingerprints_month01, locations_month01 = get_all_data_month(db_path, "01", 15, 5)
    fingerprints_month02, locations_month02 = get_all_data_month(db_path, "02", 1, 5)
    fingerprints_month15, locations_month15 = get_all_data_month(db_path, "15", 1, 5)

    th = -80
    pct = 0.1

    v_index_db01 = get_index_ap_with_data(fingerprints_month01, th, pct)
    v_index_db02 = get_index_ap_with_data(fingerprints_month02, th, pct)
    v_index_db15 = get_index_ap_with_data(fingerprints_month15, th, pct)

    l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db01, v_index_db02)
    print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both)))

    l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db01, v_index_db15)
    print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both)))


# -----------------------------------------------------------
# The objective of this experiment is to obtain some data about the number of AP in common along time
# -----------------------------------------------------------
def Experiment2(db_path):

    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 5)
    v_index_db1 = get_index_ap_with_data(fingerprints_month1)

    data = np.zeros([14,3])
    for i in range(2,16):
        month_str = "{0:02d}".format(i)
        fingerprints_monthi, locations_monthi = get_all_data_month(db_path, month_str, 1, 5)
        v_index_dbi = get_index_ap_with_data(fingerprints_monthi)
        l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db1, v_index_dbi)
        data[i-2,] = [len(l_both), len(l_new), len(l_gone)]

    print(data)
    plt.plot(data[:, 0])
    plt.plot(data[:, 1])
    plt.plot(data[:, 2])
    plt.legend(['Both', 'New', 'Gone'], loc='upper left')
    plt.show()


# -----------------------------------------------------------
# The objective of this experiment is to obtain the location error where train is month 1, and test are i-th months
# -----------------------------------------------------------
def Experiment3(db_path):
    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 5)
    v_index_db1 = get_index_ap_with_data(fingerprints_month1)

    data = np.zeros([14,1])
    for i in range(2, 16):
        month_str = "{0:02d}".format(i)
        print("Month: " + month_str)

        fingerprints_monthi, locations_monthi = get_all_data_month(db_path, month_str, 1, 5)
        v_index_dbi = get_index_ap_with_data(fingerprints_monthi)
        l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db1, v_index_dbi)
        data[i-2,0] = go_common_approach(fingerprints_month1, fingerprints_monthi, l_both, locations_month1, locations_monthi)
        print(data[i-2,0])

    print(data)
    plt.plot(data[:,0])
    plt.show()

# -----------------------------------------------------------
# The objective of this experiment is to obtain the location error where train is month 1, and test are i-th months
# But only strong AP are used
# -----------------------------------------------------------
def Experiment5(db_path):
    th = -80
    pct = 0.1
    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 5)
    v_index_db1 = get_index_ap_with_data(fingerprints_month1, th, pct)

    data = np.zeros([14, 1])
    for i in range(2, 16):
        month_str = "{0:02d}".format(i)
        print("Month: " + month_str)

        fingerprints_monthi, locations_monthi = get_all_data_month(db_path, month_str, 1, 5)
        v_index_dbi = get_index_ap_with_data(fingerprints_monthi, th, pct)
        l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db1, v_index_dbi)
        print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both)))

        data[i - 2, 0] = go_common_approach(fingerprints_month1, fingerprints_monthi, l_both, locations_month1, locations_monthi)
        print(data[i - 2, 0])

    print(data)
    plt.plot(data[:, 0])
    plt.show()

# -----------------------------------------------------------
# The same than experiments 6 but now only train samples are used in the month1 (training) and test ones in the rest.
# Train samples were captured in different placed than the test ones
# -----------------------------------------------------------
def Experiment6(db_path):
    th = -80
    pct = 0.1
    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 0)
    v_index_db1 = get_index_ap_with_data(fingerprints_month1, th, pct)

    data = np.zeros([14, 1])
    for i in range(2, 16):
        month_str = "{0:02d}".format(i)
        print("Month: " + month_str)

        fingerprints_monthi, locations_monthi = get_all_data_month(db_path, month_str, 0, 5)
        v_index_dbi = get_index_ap_with_data(fingerprints_monthi, th, pct)
        l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db1, v_index_dbi)
        print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both)))

        data[i - 2, 0] = go_common_approach(fingerprints_month1, fingerprints_monthi, l_both, locations_month1,
                                                locations_monthi)
        print(data[i - 2, 0])

    print(data)
    plt.plot(data[:, 0])
    plt.show()


# -----------------------------------------------------------
# 1 vs 2 y 1 vs 15. Solo comunes y solo "buenas". Train con train, Test con test
# Es para comprobar si el algoritmo de posicionamiento funciona o no bien
# -----------------------------------------------------------
def Experiment7(db_path):
    th = -80
    pct = 0.1
    fingerprints_month01, locations_month01 = get_all_data_month(db_path, "01", 15, 0)
    v_index_db01 = get_index_ap_with_data(fingerprints_month01, th, pct)

    fingerprints_month02, locations_month02 = get_all_data_month(db_path, "02", 0, 5)
    v_index_db02 = get_index_ap_with_data(fingerprints_month02, th, pct)

    fingerprints_month15, locations_month15 = get_all_data_month(db_path, "15", 0, 5)
    v_index_db15 = get_index_ap_with_data(fingerprints_month15, th, pct)

    l_new, l_gone, l_both02 = get_new_gone_both_aps(v_index_db01, v_index_db02)
    print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both02)))
    print(l_both02)

    l_new, l_gone, l_both15 = get_new_gone_both_aps(v_index_db01, v_index_db15)
    print(str(len(l_new)) + " " + str(len(l_gone)) + " " + str(len(l_both15)))
    print(l_both15)

    acc = go_common_approach(fingerprints_month01, fingerprints_month02, l_both02, locations_month01, locations_month02, "02",l_both02)
    print("01 vs 02 -> " + str(acc))

    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both15, locations_month01, locations_month15, "15",l_both15)
    print("01 vs 15 -> " + str(acc))

# -----------------------------------------------------------
def Experiment8():
    trainFP02 = pd.read_csv("trainFP02.csv")
    testFP02 = pd.read_csv("testFP02.csv")


    trainFP15 = pd.read_csv("trainFP15.csv")
    testFP15 = pd.read_csv("testFP15.csv")


    pct_train_02 = np.sum(trainFP02!=0)/trainFP02.shape[0]
    pct_test_02 = np.sum(testFP02!=0)/testFP02.shape[0]

    print(pct_train_02)
    print(pct_test_02)

    pct_train_15 = np.sum(trainFP15!=0)/trainFP15.shape[0]
    pct_test_15 = np.sum(testFP15!=0)/testFP15.shape[0]

    print(pct_train_15)
    print(pct_test_15)

# -----------------------------------------------------------
# Calcular la maxima distancia que existe entre dos puntos de la BD
# -----------------------------------------------------------
def Experiment9():
    fingerprints_month15, locations_month15 = get_all_data_month(db_path, "15", 1, 5)

    distances = distance.cdist(locations_month15[:,1:2], locations_month15[:,1:2], "euclidean")
    print(np.max(distances))

# -----------------------------------------------------------
# Uso en l_both propio. 01 vs 15
# -----------------------------------------------------------
def Experiment10():
    fingerprints_month01, locations_month01 = get_all_data_month(db_path, "01", 15, 0)
    fingerprints_month15, locations_month15 = get_all_data_month(db_path, "15", 0, 5)

    l_both_org =[16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    l_both_org =[5,8, 16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    l_both_org =[6,8,16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    l_both_org =[7,8,16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    l_both_org =[11,8,16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    l_both_org =[21,8,16,50,51,70]
    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both_org, locations_month01, locations_month15)
    print(str(l_both_org) + "->" + str(acc))

    #quintando uno de la lista
    #l_both_org =[16,21,50,51,70]
    #l_both = list(l_both_org)
    #for l in l_both_org:
    #    l_both.remove(l)
    #    acc = go_common_approach(fingerprints_month01, fingerprints_month15, l_both, locations_month01, locations_month15)
    #    print (str(l_both) + "->" + str(acc))
    #    l_both = list(l_both_org)






"""
    fingerprints_month1, locations_month1 = get_all_data_month(db_path, "01", 15, 5)
    fingerprints_month2, locations_month2 = get_all_data_month(db_path, "15", 1, 5)

    v_index_db1 = get_index_ap_with_data(fingerprints_month1)
    v_index_db2 = get_index_ap_with_data(fingerprints_month2)
    l_new, l_gone, l_both = get_new_gone_both_aps(v_index_db1, v_index_db2)

    print("DB 1 has " + str(v_index_db1.sum()))
    print("DB 2 has " + str(v_index_db2.sum()))
    print("APs in both [" + str(len(l_both)) + "] ->")
    print(l_both)
    print("New APs in DB2 [" + str(len(l_new)) + "] ->")
    print(l_new)
    print("APs in DB1 but not in BD2 [" + str(len(l_gone)) + "] ->")
    print(l_gone)

    accuracy_common = go_common_approach(fingerprints_month1, fingerprints_month2, l_both, locations_month1, locations_month2)
    accuracy_proposed = go_proposed_approach(fingerprints_month1, fingerprints_month2, l_new, l_gone, locations_month1, locations_month2)
