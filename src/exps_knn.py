import argparse
import json
import os
import numpy as np
import pandas as pd
from utils.data import load_dataset
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
from time import time
from algos.dilca import Dilca
from algos.dilca_fair import DilcaFair
from utils.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio, equal_opportunity_difference, equal_opportunity_ratio

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="FairDILCA KNN experiments")
    parser.add_argument("dataset_name", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("n_bins", type=int, help="Number of bins to be used by discretization technique")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    
    args = parser.parse_args()

    dname = args.dataset_name
    max_size = args.max_size # max number of records per dataset
    n_bins = args.n_bins
    datasets_path = "rsc/originals/"
    seed = 0

    if not os.path.exists("out/results"): os.mkdir("out/results")

    # load datasets info
    f = open("rsc/datasets.json")
    datasets = json.load(f)
    f.close()
    
    # settings
    ks_knn = [5, 7, 11, 15, 23]
    contexts = [('M', 'FM'), ('RR', 'FRR1'), ('RR', 'FRR2')]
    alphas = [round(a, 2) for a in np.arange(0.0, 1.0, 0.05)]
    cols = ['dataset', 'objs_no', 'n_bins', 'alpha', 'k','sensitive', 'fair_method', 'method', 'empty_context', 'sigma', "mode", "accuracy_nf", "accuracy_f", "f1_macro_nf", "f1_macro_f", "mcc_nf", "mcc_f", "demographic_parity_diff_nf", "demographic_parity_diff_f", "equalized_odds_diff_nf", "equalized_odds_diff_f", "equal_opportunity_diff_nf", "equal_opportunity_diff_f", "demographic_parity_ratio_nf", "demographic_parity_ratio_f", "equalized_odds_ratio_nf", "equalized_odds_ratio_f", "equal_opportunity_ratio_nf", "equal_opportunity_ratio_f", "time"]
    
    datasets_path + "{}.data".format(dname) # data
    results_d_path = "out/results/{}".format(dname) # storing results for dataset d
    checkpoint_path = "out/results/exps_{}_knn_checkpoint.csv".format(dname) # checkpoint for dataset d
        
    if not os.path.exists(results_d_path): os.mkdir(results_d_path)
    
    results_df = pd.DataFrame(columns=cols)
    exp_no = 1
    checkpoint_exists = False
    dataset, X, Y = load_dataset(datasets[dname], datasets_path)
    if max_size != None:
        dataset = dataset.sample(frac=1, random_state=seed)
        X = X.sample(frac=1, random_state=seed)
        Y = Y.sample(frac=1, random_state=seed)
        X = X[:max_size]
        Y = Y[:max_size]
        dataset = dataset[:max_size]
    dataset.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    missing_values = None
    if "mv_label" in datasets[dname]: missing_values = datasets[dname]["mv_label"]

    k = Y.unique().shape[0]

    if os.path.exists(checkpoint_path):
            results_df = pd.read_pickle(checkpoint_path)
            from_exp = len(results_df)
            checkpoint_exists = True
            print("Checkpoint containing {} tests loaded.".format(from_exp))

    sensitives = [int(s) for s in list(datasets[dname]['sensitives'].keys())]
    pos_label = datasets[dname]['pos_label']
    
    # preprocess dataset
    model = Dilca(method = 'M', sigma = 1, missing_values = missing_values, mv_handling='mean_mode', discretize = 'kmeans',n_bins = n_bins, dtypes = 'keep')
    model._init_all(X, y = None)
    Xp = model._dataset.copy()
    
    # train/test splitting see function check_dataset in algos/dilca.py
    c = True
    rs = seed -1
    for i in range(10):
        if c:
            X_train, X_test, y_train, y_test = train_test_split(Xp, Y, test_size = .3, stratify = Y, random_state = rs + 1)
            c = model.check_dataset(X_test, train_dataset = X_train)
            rs += 1
    if c: raise ValueError('Unable to correctly split the dataset')
    for k_knn in ks_knn:
        for m, fm in contexts:
            # RUN NON FAIR DILCA
            model_nf = Dilca(preprocessing=False, method=m, sigma=1.0)
            model_nf.fit(X_train)
            new_test_nf = model_nf.encoding(dataset = X_test)
            new_train_nf = model_nf.encoding(dataset = X_train)
            # TRAIN KNN MODEL
            knn_model = knn(k_knn, metric = 'euclidean')
            knn_model.fit(new_train_nf, y_train)
            labels_nf = knn_model.predict(new_test_nf)
            # PERFORMANCE/ACCURACY
            accuracy_nf = accuracy_score(y_test, labels_nf)
            macro_f1_nf = f1_score(y_test, labels_nf, average='macro')
            mcc_nf = mcc(y_test, labels_nf)

            print("\nNON FAIR MODEL STATS\naccuracy: {}\tf1 (macro): {}\t MCC: {}".format(accuracy_nf, macro_f1_nf, mcc_nf))
            
            for s in sensitives:
                # FAIRNESS METRICS (DILCA)
                group_membership = X[s][X_test.index.values] 
                dp_diff_nf = demographic_parity_difference(labels_nf, pos_label, group_membership)
                dp_ratio_nf = demographic_parity_ratio(labels_nf, pos_label, group_membership)
                eodds_diff_nf = equalized_odds_difference(y_test, labels_nf, pos_label, group_membership)
                eodds_ratio_nf = equalized_odds_ratio(y_test, labels_nf, pos_label, group_membership)
                eopp_diff_nf = equal_opportunity_difference(y_test, labels_nf, pos_label, group_membership)
                eopp_ratio_nf = equal_opportunity_ratio(y_test, labels_nf, pos_label, group_membership)
                
                frr2_flag=False
                for a in alphas:
                    if frr2_flag: continue
                    if checkpoint_exists:
                        if exp_no <= from_exp:
                            if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                            if results_df['alpha'][exp_no] != a: raise ValueError("Experiment config is not the same: alpha")
                            if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                            if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                            #if results_df['clustering_type'][exp_no] != ct: raise ValueError("Experiment config is not the same: clustering_type")
                            if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                            exp_no += 1
                            continue
                    print(dname, "# {}\t (alpha: {}, fair_method: {}, sensitive: {} ({}))".format(exp_no, a, fm, datasets[dname]['sensitives'][str(s)], s))
                    ec = False
                    start_time = time()
                    
                    model_f = DilcaFair(s, alpha=a, fair_method=fm, method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values, preprocessing=False)
                    model_f.fit(X_train)
                    
                    try:
                        model_f.fit_fair(X_train)
                        new_test_f = model_f.encoding_fair(dataset = X_test, distance_list=model_f._distance_list_fair)
                        new_train_f = model_f.encoding_fair(dataset = X_train, distance_list=model_f._distance_list_fair)
                        # TRAIN FAIR KNN
                        knn_model_f = knn(5, metric = 'euclidean')
                        knn_model_f.fit(new_train_f, y_train)
                        labels_f = knn_model_f.predict(new_test_f)
                        # ACCURACY
                        accuracy_f = accuracy_score(y_test, labels_f)
                        macro_f1_f = f1_score(y_test, labels_f, average='macro')
                        mcc_f = mcc(y_test, labels_f)
                        
                        print("accuracy: {}\tf1 (macro): {}\t MCC: {}\n".format(accuracy_f, macro_f1_f, mcc_f))
                        
                        # FAIRNESS METRICS (FairDILCA)
                        dp_diff_f = demographic_parity_difference(labels_f, pos_label, group_membership)
                        dp_ratio_f = demographic_parity_ratio(labels_f, pos_label, group_membership)
                        eodds_diff_f = equalized_odds_difference(y_test, labels_f, pos_label, group_membership)
                        eodds_ratio_f = equalized_odds_ratio(y_test, labels_f, pos_label, group_membership)
                        eopp_diff_f = equal_opportunity_difference(y_test, labels_f, pos_label, group_membership)
                        eopp_ratio_f = equal_opportunity_ratio(y_test, labels_f, pos_label, group_membership)
                        
                    except Exception as e:
                        if str(e) == "Cannot return a context.":
                            print("Empty context")
                            ec = True
                            accuracy_f = macro_f1_f = mcc_f = None
                            dp_diff_f = dp_ratio_f = eodds_diff_f = eodds_ratio_f = eopp_diff_f = eopp_ratio_f = None
                        else: raise ValueError(e)

                    end_time = time()
                    t_sec = end_time - start_time
                    results_df.loc[exp_no] = [dname, len(X), n_bins, a, k_knn, datasets[dname]['sensitives'][str(s)], fm, m, ec, 1.0, "CD", accuracy_nf, accuracy_f, macro_f1_nf, macro_f1_f, mcc_nf, mcc_f, dp_diff_nf, dp_diff_f, eodds_diff_nf, eodds_diff_f, eopp_diff_nf, eopp_diff_f, dp_ratio_nf, dp_ratio_f, eodds_ratio_nf, eodds_ratio_f, eopp_ratio_nf, eopp_ratio_f, t_sec]
                    exp_no += 1
                    results_df.to_pickle(checkpoint_path)
                    if fm == "FRR2": frr2_flag=True
    results_df.to_pickle("out/results/exps_{}_knn.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
    
    # BASELINES
    for k_knn in ks_knn:
        for m, fm in contexts:
            # RUN NON FAIR DILCA
            model_nf = Dilca(preprocessing=False, method=m, sigma=1.0)
            model_nf.fit(X_train)
            new_test_nf = model_nf.encoding(dataset = X_test)
            new_train_nf = model_nf.encoding(dataset = X_train)
            # TRAIN KNN MODEL
            knn_model = knn(5, metric = 'euclidean')
            knn_model.fit(new_train_nf, y_train)
            labels_nf = knn_model.predict(new_test_nf)
            # PERFORMANCE/ACCURACY
            accuracy_nf = accuracy_score(y_test, labels_nf)
            macro_f1_nf = f1_score(y_test, labels_nf, average='macro')
            mcc_nf = mcc(y_test, labels_nf)

            print("\nNON FAIR MODEL STATS\naccuracy: {}\tf1 (macro): {}\t MCC: {}".format(accuracy_nf, macro_f1_nf, mcc_nf))
            
            for s in sensitives:
                # FAIRNESS METRICS (DILCA)
                group_membership = X[s][X_test.index.values] 
                dp_diff_nf = demographic_parity_difference(labels_nf, pos_label, group_membership)
                dp_ratio_nf = demographic_parity_ratio(labels_nf, pos_label, group_membership)
                eodds_diff_nf = equalized_odds_difference(y_test, labels_nf, pos_label, group_membership)
                eodds_ratio_nf = equalized_odds_ratio(y_test, labels_nf, pos_label, group_membership)
                eopp_diff_nf = equal_opportunity_difference(y_test, labels_nf, pos_label, group_membership)
                eopp_ratio_nf = equal_opportunity_ratio(y_test, labels_nf, pos_label, group_membership)
                
                frr2_flag=False
                for a in alphas:
                    if frr2_flag: continue
                    if checkpoint_exists:
                        if exp_no <= from_exp:
                            if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                            if results_df['alpha'][exp_no] != a: raise ValueError("Experiment config is not the same: alpha")
                            if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                            if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                            #if results_df['clustering_type'][exp_no] != ct: raise ValueError("Experiment config is not the same: clustering_type")
                            if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                            exp_no += 1
                            continue
                    print(dname, "# {}\t (alpha: {}, fair_method: {}, sensitive: {} ({}))".format(exp_no, a, fm, datasets[dname]['sensitives'][str(s)], s))
                    ec = False
                    start_time = time()
                    
                    model_f = DilcaFair(s, alpha=a, fair_method=fm, method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values, preprocessing=False, distance_fairness=False)
                    model_f.fit(X_train)
                    
                    try:
                        model_f.fit_fair(X_train)
                        new_test_f = model_f.encoding_fair(dataset = X_test, distance_list=model_f._distance_list_fair)
                        new_train_f = model_f.encoding_fair(dataset = X_train, distance_list=model_f._distance_list_fair)
                        # TRAIN FAIR KNN
                        knn_model_f = knn(5, metric = 'euclidean')
                        knn_model_f.fit(new_train_f, y_train)
                        labels_f = knn_model_f.predict(new_test_f)
                        # ACCURACY
                        accuracy_f = accuracy_score(y_test, labels_f)
                        macro_f1_f = f1_score(y_test, labels_f, average='macro')
                        mcc_f = mcc(y_test, labels_f)
                        
                        print("accuracy: {}\tf1 (macro): {}\t MCC: {}\n".format(accuracy_f, macro_f1_f, mcc_f))
                        
                        # FAIRNESS METRICS (FairDILCA)
                        dp_diff_f = demographic_parity_difference(labels_f, pos_label, group_membership)
                        dp_ratio_f = demographic_parity_ratio(labels_f, pos_label, group_membership)
                        eodds_diff_f = equalized_odds_difference(y_test, labels_f, pos_label, group_membership)
                        eodds_ratio_f = equalized_odds_ratio(y_test, labels_f, pos_label, group_membership)
                        eopp_diff_f = equal_opportunity_difference(y_test, labels_f, pos_label, group_membership)
                        eopp_ratio_f = equal_opportunity_ratio(y_test, labels_f, pos_label, group_membership)
                        
                    except Exception as e:
                        if str(e) == "Cannot return a context.":
                            print("Empty context")
                            ec = True
                            accuracy_f = macro_f1_f = mcc_f = None
                            dp_diff_f = dp_ratio_f = eodds_diff_f = eodds_ratio_f = eopp_diff_f = eopp_ratio_f = None
                        else: raise ValueError(e)

                    end_time = time()
                    t_sec = end_time - start_time
                    results_df.loc[exp_no] = [dname, len(X), n_bins, a, k_knn, datasets[dname]['sensitives'][str(s)], fm, m, ec, 1.0, "C", accuracy_nf, accuracy_f, macro_f1_nf, macro_f1_f, mcc_nf, mcc_f, dp_diff_nf, dp_diff_f, eodds_diff_nf, eodds_diff_f, eopp_diff_nf, eopp_diff_f, dp_ratio_nf, dp_ratio_f, eodds_ratio_nf, eodds_ratio_f, eopp_ratio_nf, eopp_ratio_f, t_sec]
                    exp_no += 1
                    results_df.to_pickle(checkpoint_path)
                    if fm == "FRR2": frr2_flag=True
    results_df.to_pickle("out/results/exps_{}_knn.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
                
if __name__ == "__main__":
    main()