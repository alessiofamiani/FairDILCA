import os
import json
import argparse
import numpy as np
import pandas as pd
from utils.data import load_dataset, handle_mv
from utils.metrics import bcss, f_score, max_fairness_cost, balance_entropy, balance_gen, demographic_parity_difference, demographic_parity_ratio, equal_opportunity_difference, equal_opportunity_ratio, equalized_odds_difference, equalized_odds_ratio
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from algos.dilca import Dilca
from fairlearn.preprocessing import CorrelationRemover
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from time import time
from exps import plot_tsne
from exps_clustering import run_clustering_exp
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef as mcc

from utils.metrics import compute_groups_means

import warnings
warnings.filterwarnings("ignore")


SENSITIVE_COLORS = [plt.cm.Dark2(i) for i in range(8)]
TARGET_COLORS = [plt.cm.tab10(i) for i in range(10)]

# ------------------------------------
#        CORRELATION REMOVER
# ------------------------------------
def cr_exps_knn(X_train, X_test, y_train, y_test, k, pos_label, group_membership, sensitive=[0], alpha=0.0):
    start_time = time()
    # 1) Apply CR on train and test
    cr = CorrelationRemover(sensitive_feature_ids=sensitive, alpha=alpha)
    cr.fit(X_train)
    new_train_cr = cr.transform(X_train)
    new_train_cr = pd.DataFrame(data=new_train_cr, columns=[i for i in range(new_train_cr.shape[1])])
    new_test_cr = cr.transform(X_test)
    new_test_cr = pd.DataFrame(data=new_test_cr, columns=[i for i in range(new_test_cr.shape[1])])

    # 2) KNN on CR
    knn_cr = knn(k, metric = 'euclidean')
    knn_cr.fit(new_train_cr, y_train)
    labels_cr = knn_cr.predict(new_test_cr)
    
    # 3) Accuracy (CR)
    accuracy_cr = accuracy_score(y_test, labels_cr)
    macro_f1_cr = f1_score(y_test, labels_cr, average='macro')
    mcc_cr = mcc(y_test, labels_cr)
    
    # Fairness (CR)
    dp_diff_cr = demographic_parity_difference(labels_cr, pos_label, group_membership)
    dp_ratio_cr = demographic_parity_ratio(labels_cr, pos_label, group_membership)
    eodds_diff_cr = equalized_odds_difference(y_test, labels_cr, pos_label, group_membership)
    eodds_ratio_cr = equalized_odds_ratio(y_test, labels_cr, pos_label, group_membership)
    eopp_diff_cr = equal_opportunity_difference(y_test, labels_cr, pos_label, group_membership)
    eopp_ratio_cr = equal_opportunity_ratio(y_test, labels_cr, pos_label, group_membership)
    
    end_time = time()
    t_sec_cr = end_time - start_time
    
    return labels_cr, accuracy_cr, macro_f1_cr, mcc_cr, dp_diff_cr, dp_ratio_cr, eodds_diff_cr, eodds_ratio_cr, eopp_diff_cr, eopp_ratio_cr, t_sec_cr

def cr_exps_clustering(X, labels_dilca, k, ct, original_dataset, s, sensitive=[0], alpha=0.0):
    start_time = time()
            
    # 1. Apply CR to (preprocessed) categorical data        
    cr = CorrelationRemover(sensitive_feature_ids=sensitive, alpha=alpha)
    cr.fit(X)
    new_cr = cr.transform(X)
    new_cr = pd.DataFrame(data=new_cr, columns=[i for i in range(new_cr.shape[1])])
    
    # 2. Clustering on the new representation of CR w.r.t. DILCA labels
    ari_cr, nmi_cr, sil_cr, labels_cr = run_clustering_exp(new_cr, k, labels_dilca, clustering_type=ct)
    mfc_clusters_cr = max_fairness_cost(original_dataset, s, labels_cr)
    generalised_balance_cr = balance_gen(original_dataset, s, labels_cr)
    entropy_balance_cr = balance_entropy(original_dataset, s, labels_cr)
    
    end_time = time()
    t_sec = end_time - start_time
    return labels_cr, ari_cr, nmi_cr, sil_cr, mfc_clusters_cr, generalised_balance_cr, entropy_balance_cr, t_sec

def cr_exps(X, Y, exp_no, datasets, original_dataset, s, objs_distances_dilca, objs_distances_original, sensitive=[0], alpha=0.0, tsne=None, dataset="dataset"):
    start_time = time()
            
    # 1. Apply CR to (preprocessed) categorical data        
    cr = CorrelationRemover(sensitive_feature_ids=sensitive, alpha=alpha)
    cr.fit(X)
    new_cr = cr.transform(X)
    new_cr = pd.DataFrame(data=new_cr, columns=[i for i in range(new_cr.shape[1])])
    
    z = None
    # 2. t-SNE
    if tsne != None:
        z = tsne.fit_transform(new_cr)
        labels_sensitive = original_dataset[s]
        fig_sensitive, ax_sensitive = plt.subplots(figsize=(15, 10), dpi=300)
        fig_target, ax_target = plt.subplots(figsize=(15, 10), dpi=300)
        plot_tsne(z, labels_sensitive, ax_sensitive, SENSITIVE_COLORS)
        plot_tsne(z, Y, ax_target, TARGET_COLORS)
        fig_sensitive.tight_layout()
        fig_target.tight_layout()
        fig_sensitive.savefig("out/results/competitors/{}/tsne_CR_s{}_a{}.png".format(dataset, datasets[dataset]['sensitives'][str(s)], alpha), bbox_inches='tight', dpi=300)
        fig_target.savefig("out/results/competitors/{}/tsne_CR_s{}_a{}_target.png".format(dataset, datasets[dataset]['sensitives'][str(s)], alpha), bbox_inches='tight', dpi=300)
        plt.close(fig_sensitive)
        plt.close(fig_target)

    # 3. Getting objs pairwise distances via euclidean()
    objs_distances_cr = euclidean_distances(new_cr)
    
    # 3.1 Computing pearson between DILCA objs distances and CR objs distances
    flatten_cr = objs_distances_cr[np.triu_indices(objs_distances_cr.shape[0], k=1)]
    flatten_dilca = objs_distances_dilca[np.triu_indices(objs_distances_dilca.shape[0], k=1)]
    flatten_original = objs_distances_original[np.triu_indices(objs_distances_original.shape[0], k=1)]
                    
    p_dilca = pearsonr(flatten_cr, flatten_dilca)[0]
    p_original = pearsonr(flatten_cr, flatten_original)[0]
    
    # 3.2 Obtaining means  
    means_cr, means_MaG_Migs_cr, groups, group_sizes, majority_group = compute_groups_means(original_dataset, s, objs_distances_cr)
    means_cr_nn, means_MaG_Migs_cr_nn, groups, group_sizes, majority_group = compute_groups_means(original_dataset, s, objs_distances_cr, normalise=False)
    end_time = time()
    t_sec = end_time - start_time

    return means_cr, means_MaG_Migs_cr, means_cr_nn, means_MaG_Migs_cr_nn, t_sec, p_dilca, p_original, objs_distances_cr, z

# ------------------------------------
#             EXPERIMENTS
# ------------------------------------

def main():
    # FIXME: tsne_flag argparse
    parser = argparse.ArgumentParser(description="Competitors experiments")
    parser.add_argument("competitor", type=str, choices=["CorrelationRemover"], help="Name of the competitor to test")
    parser.add_argument("exp_type", type=str, choices=["exps", "knn", "clustering"], help="Type of experiments")
    parser.add_argument("dataset_name", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("n_bins", type=int, help="Number of bins to be used by discretization technique")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    parser.add_argument("-tsne", "--tsne_flag", choices=['True', 'False'], help="Flag for tsne computation", default=False)
    
    args = parser.parse_args()
    
    competitor = args.competitor
    exp_type = args.exp_type
    dname = args.dataset_name
    max_size = args.max_size # max number of records per dataset
    n_bins = args.n_bins
    tsne_flag = bool(args.tsne_flag)
    
    datasets_path = "rsc/originals/"
    competitors_path = "out/results/competitors"
    seed = 0

    # Create folders for outputs
    if not os.path.exists("out/results"): os.mkdir("out/results")
    if not os.path.exists(competitors_path): os.mkdir(competitors_path)
    if not os.path.exists(competitors_path + "/{}".format(dname)): os.mkdir(competitors_path + "/{}".format(dname))
    
    # load datasets info
    f = open("rsc/datasets.json")
    datasets = json.load(f)
    f.close()
    
    datasets_path + "{}.data".format(dname) # data
    results_d_path = competitors_path + "/{}".format(dname) # storing results for dataset d

    checkpoint_path = competitors_path + "/{}_{}_{}_checkpoint.result".format(exp_type, dname, competitor) # checkpoint for dataset d
    results_path = competitors_path + "/{}_{}_{}.result".format(exp_type, dname, competitor)
    results_csv_path = competitors_path + "/{}_{}_{}.csv".format(exp_type, dname, competitor)
    
    contexts = ['M', 'RR']
    alphas = [round(a, 2) for a in np.arange(0.0, 1.0, 0.05)]
    clustering_types = ["Agglomerative", "HDBSCAN", "Spectral"]
    
    exp_no = 1
    checkpoint_exists = False
    
    # Load dataset + dataset info
    dataset, X, Y = load_dataset(datasets[dname], datasets_path)
    if max_size != None:
            dataset = dataset.sample(frac=1, random_state=seed)
            X = X.sample(frac=1, random_state=seed)
            Y = Y.sample(frac=1, random_state=seed)
            X = X[:max_size]
            Y = Y[:max_size]
    dataset.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    Y.reset_index(drop=True, inplace=True)
    missing_values = None
    if "mv_label" in datasets[dname]: missing_values = datasets[dname]["mv_label"]
    sensitives = [int(s) for s in datasets[dname]["sensitives"].keys()]
    k = Y.unique().shape[0]
    pos_label = datasets[dname]['pos_label']
    
    # Check if checkpoint exists
    if os.path.exists(checkpoint_path):
        results_df = pd.read_pickle(checkpoint_path)
        from_exp = len(results_df)
        checkpoint_exists = True
        print("Checkpoint containing {} tests loaded.".format(from_exp))

    # Retrieve categorical data columns indices
    categorical_data_indices = X.select_dtypes(include=['category', 'object']).columns
    # Imputation
    Xn = handle_mv(X.copy(), mv=missing_values)
    # One-Hot Encoding of categorical features (for competitors that does not support categorical data)
    Xne = pd.get_dummies(Xn, columns=categorical_data_indices, dtype=float)
    Xne.columns = Xne.columns.astype(str)
    
    # 0. Binning, in order to produce purely categorical data
    # Using the built-in DILCA preprocessing function (ignore the method, sigma and other DILCA-related parameters)
    prep_model = Dilca(mv_handling='mean_mode', discretize = 'kmeans',n_bins = n_bins, dtypes = 'keep', method = 'M', sigma = 1.0, missing_values = missing_values,)
    prep_model._init_all(X, y = None)
    Xp = prep_model._dataset.copy()
    
    # ------------------------------------
    #       PRELIMINARY EXPERIMENTS
    # ------------------------------------
    if exp_type == "exps":
        if tsne_flag:
            tsne = TSNE(n_components=2, verbose=False, random_state=seed)
            for s in sensitives:
                print("t-SNE for {} - sensitive: {}".format(dname, datasets[dname]['sensitives'][str(s)]))
                z = tsne.fit_transform(Xne)
                labels_sensitive = X[s]
                fig_sensitive, ax_sensitive = plt.subplots(figsize=(15, 10), dpi=300)
                fig_target, ax_target = plt.subplots(figsize=(15, 10), dpi=300)
                plot_tsne(z, labels_sensitive, ax_sensitive, SENSITIVE_COLORS)
                plot_tsne(z, Y, ax_target, TARGET_COLORS)
                fig_sensitive.tight_layout()
                fig_target.tight_layout()
                fig_sensitive.savefig("out/results/competitors/{}/tsne_original_{}_s{}.png".format(dname, dname, datasets[dname]['sensitives'][str(s)]), bbox_inches='tight', dpi=300)
                fig_target.savefig("out/results/competitors/{}/tsne_original_{}_s{}_target.png".format(dname, dname, datasets[dname]['sensitives'][str(s)]), bbox_inches='tight', dpi=300)
                plt.close(fig_sensitive)
                plt.close(fig_target)
        else: tsne = None
        #results_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'groups', 'majority_group', 'z_tsne'] + ['means_competitor', 'means_dilca', 'time_competitor', 'time_dilca', 'pearson_objs', 'objs_distances_competitor', 'objs_distances_dilca']
        
        results_csv_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'groups', 'majority_group', 'groups_sizes'] + ['means_competitor', 'means_MaG-Migs_competitor', 'means_competitor_nn', 'means_MaG-Migs_competitor_nn', 'means_dilca', 'means_MaG-Migs_dilca', 'means_dilca_nn', 'means_MaG-Migs_dilca_nn', 'time_competitor', 'time_dilca', 'pearson_objs_dilca', 'pearson_objs_original']
        results_csv_df = pd.DataFrame(columns=results_csv_cols)
        results_df = pd.DataFrame(columns=results_csv_cols)
        
        start_time = time()
        objs_distances_dilca = euclidean_distances(Xne)
        
        for s in sensitives:
            # Means DILCA
            means_dilca, means_MaG_Migs_dilca, groups, group_sizes, majority_group = compute_groups_means(X, s, objs_distances_dilca)
            means_dilca_nn, means_MaG_Migs_dilca_nn, groups, group_sizes, majority_group = compute_groups_means(X, s, objs_distances_dilca, normalise=False)
            
            end_time = time()
            t_sec_dilca = end_time - start_time
            
            if competitor=="CorrelationRemover":
                # Getting all the indices of s (since it was OH encoded)
                objs_distances_original = euclidean_distances(Xne)
                sensitive_column_indices = list(Xne.filter(like="{}_".format(s)).columns)
                for a in alphas:
                    competitor_params = {"alpha": a}
                    if checkpoint_exists:
                        if exp_no <= from_exp:
                            if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                            if results_df['competitor_params'][exp_no] != competitor_params: raise ValueError("Experiment config is not the same: competitor params")
                            if results_df['n_bins'][exp_no] != n_bins: raise ValueError("Experiment config is not the same: n_bins")
                            if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                            if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                            exp_no += 1
                            continue
                    print("{} # {} {} {}\t sensitive: {} ({})".format(dname, exp_no, competitor, competitor_params, datasets[dname]['sensitives'][str(s)], s))
                    means_competitor, means_MaG_Migs_competitor, means_competitor_nn, means_MaG_Migs_competitor_nn, t_sec_competitor, pearson_competitor_dilca, pearson_competitor_original, objs_distances_competitor, z_tsne_competitor = cr_exps(Xne.copy(), Y, exp_no, datasets, X, s, objs_distances_dilca, objs_distances_original, sensitive=sensitive_column_indices, alpha=a, tsne=tsne, dataset=dname)
                    
                    # results_df.loc[exp_no] = [dname, len(X), n_bins, m, datasets[dname]['sensitives'][str(s)], competitor, competitor_params, groups, majority_group, z_tsne_competitor, means_competitor, means_dilca, t_sec_competitor, t_sec_dilca, pearson_competitor_dilca, objs_distances_competitor, objs_distances_dilca]
                    results_df.loc[exp_no] = [dname, len(X), n_bins, '', datasets[dname]['sensitives'][str(s)], competitor, competitor_params, groups, majority_group, group_sizes, means_competitor, means_MaG_Migs_competitor, means_competitor_nn, means_MaG_Migs_competitor_nn, means_dilca, means_MaG_Migs_dilca, means_dilca_nn, means_MaG_Migs_dilca_nn, t_sec_competitor, t_sec_dilca, pearson_competitor_dilca, pearson_competitor_original]
                    results_csv_df.loc[exp_no] = [dname, len(X), n_bins, '', datasets[dname]['sensitives'][str(s)], competitor, competitor_params, groups, majority_group, group_sizes, means_competitor, means_MaG_Migs_competitor, means_competitor_nn, means_MaG_Migs_competitor_nn, means_dilca, means_MaG_Migs_dilca, means_dilca_nn, means_MaG_Migs_dilca_nn, t_sec_competitor, t_sec_dilca, pearson_competitor_dilca, pearson_competitor_original]
                    exp_no += 1
                    results_df.to_pickle(checkpoint_path)
    # ------------------------------------
    #       CLUSTERING EXPERIMENTS
    # ------------------------------------
    elif exp_type == "clustering":
        results_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'groups', 'majority_group', 'time_competitor', 'time_dilca', 'labels_competitor', 'labels_dilca'] + ['clustering_type', 'ari_competitor', 'ari_dilca', 'nmi_competitor', 'nmi_dilca', 'mean_silhouette_competitor', 'mean_silhouette_dilca', 'mfc_competitor', 'mfc_dilca', 'balance_competitor', 'balance_dilca', 'entropy_competitor', 'entropy_dilca']
        results_df = pd.DataFrame(columns=results_cols)
        results_csv_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'groups', 'majority_group', 'time_competitor', 'time_dilca'] + ['clustering_type', 'ari_competitor', 'ari_dilca', 'nmi_competitor', 'nmi_dilca', 'mean_silhouette_competitor', 'mean_silhouette_dilca', 'mfc_competitor', 'mfc_dilca', 'balance_competitor', 'balance_dilca', 'entropy_competitor', 'entropy_dilca']
        results_csv_df = pd.DataFrame(columns=results_csv_cols)
        
        for ct in clustering_types:
            for m in contexts:
                start_time = time()
                # Clustering on numerical data
                # Accuracy
                print("Computing Xne (clustering)")
                ari_dilca, nmi_dilca, sil_dilca, labels_dilca = run_clustering_exp(Xne, k, Y, clustering_type=ct)
                
                for s in sensitives:
                    # Fairness
                    mfc_clusters_dilca = max_fairness_cost(X, s, labels_dilca)
                    generalised_balance_dilca = balance_gen(X, s, labels_dilca)
                    entropy_balance_dilca = balance_entropy(X, s, labels_dilca)
                    end_time = time()
                    t_sec_dilca = end_time - start_time
                    # Groups info
                    groups = X[s].unique()
                    groups_indexes = [np.where(X[s] == g)[0] for g in groups]
                    group_sizes = [gi.shape[0] for gi in groups_indexes]
                    majority_group_index = np.argmax(group_sizes)
                    majority_group = groups[majority_group_index]
                    
                    if competitor=="CorrelationRemover":
                        # Getting all the indices of s (since it was OH encoded)
                        sensitive_column_indices = list(Xne.filter(like="{}_".format(s)).columns)
                        for a in alphas:
                            competitor_params = {"alpha": a}
                            if checkpoint_exists:
                                if exp_no <= from_exp:
                                    if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                                    if results_df['clustering_type'][exp_no] != ct: raise ValueError("Experiment config is not the same: clustering_type")
                                    if results_df['competitor_params'][exp_no] != competitor_params: raise ValueError("Experiment config is not the same: competitor params")
                                    if results_df['n_bins'][exp_no] != n_bins: raise ValueError("Experiment config is not the same: n_bins")
                                    if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                                    if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                                    exp_no += 1
                                    continue
                            print("{} # {} {} {} {} {}\t sensitive: {} ({}) DILCA {}".format(dname, exp_no, exp_type, ct, competitor, competitor_params, datasets[dname]['sensitives'][str(s)], s, m))
                            labels_competitor, ari_competitor, nmi_competitor, sil_competitor, mfc_clusters_competitor, generalised_balance_competitor, entropy_balance_competitor, t_sec_competitor = cr_exps_clustering(Xne.copy(), labels_dilca, k, ct, X, s, sensitive=sensitive_column_indices, alpha=a)
                            results_df.loc[exp_no] = [dname, len(X), n_bins, m, datasets[dname]['sensitives'][str(s)], competitor, competitor_params, groups, majority_group, t_sec_competitor, t_sec_dilca, labels_competitor, labels_dilca, ct, ari_competitor, ari_dilca, nmi_competitor, nmi_dilca, sil_competitor, sil_dilca, mfc_clusters_competitor, mfc_clusters_dilca, generalised_balance_competitor, generalised_balance_dilca, entropy_balance_competitor, entropy_balance_dilca]
                            results_csv_df.loc[exp_no] = [dname, len(X), n_bins, m, datasets[dname]['sensitives'][str(s)], competitor, competitor_params, groups, majority_group, t_sec_competitor, t_sec_dilca, ct, ari_competitor, ari_dilca, nmi_competitor, nmi_dilca, sil_competitor, sil_dilca, mfc_clusters_competitor, mfc_clusters_dilca, generalised_balance_competitor, generalised_balance_dilca, entropy_balance_competitor, entropy_balance_dilca]
                            exp_no += 1
                            results_df.to_pickle(checkpoint_path)
    # ------------------------------------
    #          KNN EXPERIMENTS
    # ------------------------------------
    elif exp_type == "knn":
        ks_knn = [5, 7, 11, 15, 23]
        results_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'time_competitor', 'time_dilca', 'k', 'labels_competitor', 'labels_dilca'] + ['accuracy_competitor', 'accuracy_dilca', 'f1_macro_competitor', 'f1_macro_dilca', 'mcc_competitor', 'mcc_dilca', 'demographic_parity_diff_competitor', 'demographic_parity_diff_dilca', 'equalized_odds_diff_competitor', 'equalized_odds_diff_dilca', 'equal_opportunity_diff_competitor', 'equal_opportunity_diff_dilca', 'demographic_ratio_competitor', 'demographic_ratio_dilca', 'equalized_odds_ratio_competitor', 'equalized_odds_ratio_dilca', 'equal_opportunity_ratio_competitor', 'equal_opportunity_ratio_dilca']
        results_df = pd.DataFrame(columns=results_cols)

        results_csv_cols = ['dataset', 'objs_no', 'n_bins', 'method', 'sensitive', 'competitor', 'competitor_params', 'time_competitor', 'time_dilca', 'k'] + ['accuracy_competitor', 'accuracy_dilca', 'f1_macro_competitor', 'f1_macro_dilca', 'mcc_competitor', 'mcc_dilca', 'demographic_parity_diff_competitor', 'demographic_parity_diff_dilca', 'equalized_odds_diff_competitor', 'equalized_odds_diff_dilca', 'equal_opportunity_diff_competitor', 'equal_opportunity_diff_dilca', 'demographic_ratio_competitor', 'demographic_ratio_dilca', 'equalized_odds_ratio_competitor', 'equalized_odds_ratio_dilca', 'equal_opportunity_ratio_competitor', 'equal_opportunity_ratio_dilca']
        results_csv_df = pd.DataFrame(columns=results_csv_cols)
        # 1. Train-test splitting
        # train/test splitting see function check_dataset in algos/dilca.py
        c = True
        rs = seed -1
        for i in range(10):
            if c:
                X_train, X_test, y_train, y_test = train_test_split(Xp, Y, test_size = .3, stratify = Y, random_state = rs + 1)
                c = prep_model.check_dataset(X_test, train_dataset = X_train)
                # For competitors 
                X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(Xne, Y, test_size = .3, stratify = Y, random_state = rs + 1)
                X_train_encoded.columns = X_train_encoded.columns.astype(str)
                X_test_encoded.columns = X_test_encoded.columns.astype(str)
                
                rs += 1
        if c: raise ValueError('Unable to correctly split the dataset')
        #print((X_train.index.values == X_train_encoded.index.values).all())
        print((y_test.index.values == y_test_encoded.index.values).all())
        for k_knn in ks_knn:
            for m in contexts:
                print("Computing Xne (knn)")
                start_time = time()      
                # KNN 
                knn_dilca = knn(k_knn, metric = 'euclidean')
                knn_dilca.fit(X_train_encoded, y_train)
                labels_dilca = knn_dilca.predict(X_test_encoded)
                # Accuracy 
                accuracy_dilca = accuracy_score(y_test, labels_dilca)
                macro_f1_dilca = f1_score(y_test, labels_dilca, average='macro')
                mcc_dilca = mcc(y_test, labels_dilca)
                
                for s in sensitives:
                    # Fairness 
                    group_membership = X[s][X_test.index.values] 
                    dp_diff_dilca = demographic_parity_difference(labels_dilca, pos_label, group_membership)
                    dp_ratio_dilca = demographic_parity_ratio(labels_dilca, pos_label, group_membership)
                    eodds_diff_dilca = equalized_odds_difference(y_test, labels_dilca, pos_label, group_membership)
                    eodds_ratio_dilca = equalized_odds_ratio(y_test, labels_dilca, pos_label, group_membership)
                    eopp_diff_dilca = equal_opportunity_difference(y_test, labels_dilca, pos_label, group_membership)
                    eopp_ratio_dilca = equal_opportunity_ratio(y_test, labels_dilca, pos_label, group_membership)
                    
                    end_time = time()
                    t_sec_dilca = end_time - start_time
                    if competitor=="CorrelationRemover":
                        # Getting all the indices of s (since it was OH encoded)
                        sensitive_column_indices = list(X_train_encoded.filter(like="{}_".format(s)).columns)
                        for a in alphas:
                            competitor_params = {"alpha": a}
                            if checkpoint_exists:
                                if exp_no <= from_exp:
                                    if results_df['k'][exp_no] != k_knn: raise ValueError("Experiment config is not the same: k")
                                    if results_df['competitor_params'][exp_no] != competitor_params: raise ValueError("Experiment config is not the same: competitor params")
                                    if results_df['n_bins'][exp_no] != n_bins: raise ValueError("Experiment config is not the same: n_bins")
                                    if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                                    if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                                    exp_no += 1
                                    continue
                        
                            print("{} # {} {} k:{} {} {}\t sensitive: {} ({}) DILCA {}".format(dname, exp_no, exp_type, k_knn, competitor, competitor_params, datasets[dname]['sensitives'][str(s)], s, m))

                            labels_competitor, accuracy_competitor, macro_f1_competitor, mcc_competitor, dp_diff_competitor, dp_ratio_competitor, eodds_diff_competitor, eodds_ratio_competitor, eopp_diff_competitor, eopp_ratio_competitor, t_sec_competitor = cr_exps_knn(X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded, k_knn, pos_label, group_membership, sensitive=sensitive_column_indices, alpha=a)
               
                            results_df.loc[exp_no] = [dname, len(X), n_bins, m, datasets[dname]['sensitives'][str(s)], competitor, competitor_params, t_sec_competitor, t_sec_dilca, k_knn, labels_competitor, labels_dilca, accuracy_competitor, accuracy_dilca, macro_f1_competitor, macro_f1_dilca, mcc_competitor, mcc_dilca, dp_diff_competitor, dp_diff_dilca, eodds_diff_competitor, eodds_diff_dilca, eopp_diff_competitor, eopp_diff_dilca, dp_ratio_competitor, dp_ratio_dilca, eodds_ratio_competitor, eodds_ratio_dilca, eopp_ratio_competitor, eopp_ratio_dilca]
                            results_csv_df.loc[exp_no] = [dname, len(X), n_bins, m, datasets[dname]['sensitives'][str(s)], competitor, competitor_params, t_sec_competitor, t_sec_dilca, k_knn, accuracy_competitor, accuracy_dilca, macro_f1_competitor, macro_f1_dilca, mcc_competitor, mcc_dilca, dp_diff_competitor, dp_diff_dilca, eodds_diff_competitor, eodds_diff_dilca, eopp_diff_competitor, eopp_diff_dilca, dp_ratio_competitor, dp_ratio_dilca, eodds_ratio_competitor, eodds_ratio_dilca, eopp_ratio_competitor, eopp_ratio_dilca]
                            exp_no += 1
                            results_df.to_pickle(checkpoint_path)
    else: raise ValueError("Experiment type not allowed.")
    results_df.to_pickle(results_path)
    results_csv_df.to_pickle(results_csv_path)
    
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
            
if __name__ == "__main__":
    main()