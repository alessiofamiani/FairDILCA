from algos.dilca_fair import DilcaFair
from algos.dilca import Dilca
from utils.metrics import max_fairness_cost, balance_entropy, balance_gen
from utils.data import load_dataset

from sklearn.cluster import AgglomerativeClustering, HDBSCAN, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari, silhouette_score

import pandas as pd
import numpy as np
import json
import os
from time import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse

def run_clustering_exp(new, k, y, clustering_type, linkage="ward"):
    if clustering_type == "Agglomerative":
        model = AgglomerativeClustering(k, linkage=linkage)
    elif clustering_type == "HDBSCAN":
        model = HDBSCAN()
    elif clustering_type == "Spectral":
        model = SpectralClustering(n_clusters=k, random_state=0)
    
    labels = model.fit_predict(new)
    ari_score = ari(y, labels)
    nmi_score = nmi(y, labels)
    sil_score = silhouette_score(new, labels)

    return ari_score, nmi_score, sil_score, labels

def main():
    parser = argparse.ArgumentParser(description="FairDILCA Clustering experiments")
    parser.add_argument("dataset_name", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("n_bins", type=int, help="Number of bins to be used by discretization technique")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    
    args = parser.parse_args()

    dname = args.dataset_name
    max_size = args.max_size # max number of records per dataset
    n_bins= args.n_bins

    f = open("rsc/datasets.json")
    datasets = json.load(f)
    f.close()

    contexts = [('M', 'FM'), ('RR', 'FRR1'), ('RR', 'FRR2')]
    clustering_types = ["Agglomerative", "HDBSCAN", "Spectral"]
    alphas = [round(a, 2) for a in np.arange(0.0, 1.0, 0.05)]
    cols = ['dataset', 'objs_no', 'n_bins', 'clustering_type', 'alpha', 'sensitive', 'fair_method', 'method', 'empty_context', 'sigma', "mode", "groups", "majority_group", "ari_f", "ari_nf", "nmi_f", "nmi_nf", "mean_silhouette_f", "mean_silhouette_nf", "time", "mfc_f", "mfc_nf", "balance_f", "balance_nf", "entropy_f", "entropy_nf"]

    datasets_path = "rsc/originals/"
    seed = 0

    d_path = datasets_path + "{}.data".format(dname) # data
    results_d_path = "out/results/{}".format(dname) # storing results for dataset d
    checkpoint_path = "out/results/exps_{}_clustering_checkpoint.csv".format(dname) # checkpoint for dataset d
        
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
    if dname == "students_dropout":
        cols_tostr = [0, 7, 17]
        for c in cols_tostr:
            X[c] = X[c].astype(str)
    missing_values = None
    if "mv_label" in datasets[dname]: missing_values = datasets[dname]["mv_label"]

    k = Y.unique().shape[0]

    if os.path.exists(checkpoint_path):
            results_df = pd.read_pickle(checkpoint_path)
            from_exp = len(results_df)
            checkpoint_exists = True
            print("Checkpoint containing {} tests loaded.".format(from_exp))

    sensitives = [int(s) for s in list(datasets[dname]['sensitives'].keys())]

    for m, fm in contexts:
        model_nf = Dilca(method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values)
        model_nf.fit(X)
        new_nf = model_nf.encoding(distance_list=model_nf._distance_list)
        
        ari_agg_nf, nmi_agg_nf, sil_agg_nf, labels_agg_nf = run_clustering_exp(new_nf, k, Y, clustering_type="Agglomerative")
        ari_spe_nf, nmi_spe_nf, sil_spe_nf, labels_spe_nf = run_clustering_exp(new_nf, k, Y, clustering_type="Spectral")
        ari_hdb_nf, nmi_hdb_nf, sil_hdb_nf, labels_hdb_nf = run_clustering_exp(new_nf, k, Y, clustering_type="HDBSCAN")
        
        for ct in clustering_types:
            for s in sensitives:
                frr2_flag=False
                for a in alphas:
                    if frr2_flag: continue
                    if checkpoint_exists:
                        if exp_no <= from_exp:
                            if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                            if results_df['alpha'][exp_no] != a: raise ValueError("Experiment config is not the same: alpha")
                            if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                            if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                            if results_df['clustering_type'][exp_no] != ct: raise ValueError("Experiment config is not the same: clustering_type")
                            if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                            exp_no += 1
                            continue
                    print(dname, "# {}\t (alpha: {}, fair_method: {}, sensitive: {} ({}), k: {}, ct: {})".format(exp_no, a, fm, datasets[dname]['sensitives'][str(s)], s, k, ct))
                    ec = False
                    start_time = time()

                    model_f = DilcaFair(s, alpha=a, fair_method=fm, method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values)

                    try:
                        model_f.fit_fair(X)
                        new_f = model_f.encoding_fair(distance_list=model_f._distance_list_fair)
                        
                        if ct=="Agglomerative": 
                            labels_nf=labels_agg_nf
                            ari_nf = ari_agg_nf
                            nmi_nf = nmi_agg_nf
                            sil_nf = sil_agg_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        elif ct == "HDBSCAN":
                            labels_nf=labels_hdb_nf
                            ari_nf = ari_hdb_nf
                            nmi_nf = nmi_hdb_nf
                            sil_nf = sil_hdb_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        elif ct =="Spectral":
                            labels_nf=labels_spe_nf
                            ari_nf = ari_spe_nf
                            nmi_nf = nmi_spe_nf
                            sil_nf = sil_spe_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        else: raise ValueError("Clustering type not handled.")
                        #mfc_nf = max(max(mfc_clusters_nf))
                        
                        ari_f, nmi_f, sil_f, labels_f = run_clustering_exp(new_f, k, labels_nf, clustering_type=ct)
                        mfc_clusters = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_f)
                        #mfc_f = max(max(mfc_clusters))
                        
                        generalised_balance_f = balance_gen(model_f._dataset, model_f.sensitive, labels_f)
                        entropy_balance_f = balance_entropy(model_f._dataset, model_f.sensitive, labels_f)
                        
                        groups = model_f._dataset[model_f.sensitive].unique()
                        groups_indexes = [np.where(model_f._dataset[model_f.sensitive] == g)[0] for g in groups]
                        group_sizes = [gi.shape[0] for gi in groups_indexes]
                        majority_group_index = np.argmax(group_sizes)
                        majority_group = groups[majority_group_index]
                        
                        del(new_f)
                    except Exception as e:
                        if str(e) == "Cannot return a context.":
                            print("Empty context")
                            ec = True
                            ari_f = nmi_f = sil_f = mfc_f = mfc_nf = None
                        else: raise ValueError(e)
                    end_time = time()
                    t_sec = end_time - start_time
                    results_df.loc[exp_no] = [dname, len(X), n_bins, ct, a, datasets[dname]['sensitives'][str(s)], fm, m, ec, 1.0, "CD", groups, majority_group, ari_f, ari_nf, nmi_f, nmi_nf, sil_f, sil_nf, t_sec, mfc_clusters, mfc_clusters_nf, generalised_balance_f, generalised_balance_nf, entropy_balance_f, entropy_balance_nf]
                    exp_no += 1
                    results_df.to_pickle(checkpoint_path)
                    if fm == "FRR2": frr2_flag=True
    results_df.to_pickle("out/results/exps_{}_clustering.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
        
    # BASELINES

    sensitives = [int(s) for s in list(datasets[dname]['sensitives'].keys())]

    for m, fm in contexts:
        model_nf = Dilca(method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values)
        model_nf.fit(X)
        new_nf = model_nf.encoding(distance_list=model_nf._distance_list)
        
        ari_agg_nf, nmi_agg_nf, sil_agg_nf, labels_agg_nf = run_clustering_exp(new_nf, k, Y, clustering_type="Agglomerative")
        ari_spe_nf, nmi_spe_nf, sil_spe_nf, labels_spe_nf = run_clustering_exp(new_nf, k, Y, clustering_type="Spectral")
        ari_hdb_nf, nmi_hdb_nf, sil_hdb_nf, labels_hdb_nf = run_clustering_exp(new_nf, k, Y, clustering_type="HDBSCAN")
        
        for ct in clustering_types:
            for s in sensitives:
                frr2_flag=False
                for a in alphas:
                    if frr2_flag: continue
                    if checkpoint_exists:
                        if exp_no <= from_exp:
                            if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                            if results_df['alpha'][exp_no] != a: raise ValueError("Experiment config is not the same: alpha")
                            if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                            if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                            if results_df['clustering_type'][exp_no] != ct: raise ValueError("Experiment config is not the same: clustering_type")
                            if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                            exp_no += 1
                            continue
                    print(dname, "# {}\t (alpha: {}, fair_method: {}, sensitive: {} ({}), k: {}, ct: {})".format(exp_no, a, fm, datasets[dname]['sensitives'][str(s)], s, k, ct))
                    ec = False
                    start_time = time()

                    model_f = DilcaFair(s, alpha=a, fair_method=fm, method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values, distance_fairness=False)

                    try:
                        model_f.fit_fair(X)
                        new_f = model_f.encoding_fair(distance_list=model_f._distance_list_fair)
                        
                        if ct=="Agglomerative": 
                            labels_nf=labels_agg_nf
                            ari_nf = ari_agg_nf
                            nmi_nf = nmi_agg_nf
                            sil_nf = sil_agg_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        elif ct == "HDBSCAN":
                            labels_nf=labels_hdb_nf
                            ari_nf = ari_hdb_nf
                            nmi_nf = nmi_hdb_nf
                            sil_nf = sil_hdb_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        elif ct =="Spectral":
                            labels_nf=labels_spe_nf
                            ari_nf = ari_spe_nf
                            nmi_nf = nmi_spe_nf
                            sil_nf = sil_spe_nf
                            mfc_clusters_nf = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_nf)
                            generalised_balance_nf = balance_gen(model_f._dataset, model_f.sensitive, labels_nf)
                            entropy_balance_nf = balance_entropy(model_f._dataset, model_f.sensitive, labels_nf)
                        else: raise ValueError("Clustering type not handled.")
                        #mfc_nf = max(max(mfc_clusters_nf))
                        
                        ari_f, nmi_f, sil_f, labels_f = run_clustering_exp(new_f, k, labels_nf, clustering_type=ct)
                        mfc_clusters = max_fairness_cost(model_f._dataset, model_f.sensitive, labels_f)
                        #mfc_f = max(max(mfc_clusters))
                        
                        generalised_balance_f = balance_gen(model_f._dataset, model_f.sensitive, labels_f)
                        entropy_balance_f = balance_entropy(model_f._dataset, model_f.sensitive, labels_f)
                        
                        groups = model_f._dataset[model_f.sensitive].unique()
                        groups_indexes = [np.where(model_f._dataset[model_f.sensitive] == g)[0] for g in groups]
                        group_sizes = [gi.shape[0] for gi in groups_indexes]
                        majority_group_index = np.argmax(group_sizes)
                        majority_group = groups[majority_group_index]
                        
                        del(new_f)
                    except Exception as e:
                        if str(e) == "Cannot return a context.":
                            print("Empty context")
                            ec = True
                            ari_f = nmi_f = sil_f = mfc_f = mfc_nf = None
                        else: raise ValueError(e)
                    end_time = time()
                    t_sec = end_time - start_time
                    results_df.loc[exp_no] = [dname, len(X), n_bins, ct, a, datasets[dname]['sensitives'][str(s)], fm, m, ec, 1.0, "C", groups, majority_group, ari_f, ari_nf, nmi_f, nmi_nf, sil_f, sil_nf, t_sec, mfc_clusters, mfc_clusters_nf, generalised_balance_f, generalised_balance_nf, entropy_balance_f, entropy_balance_nf]
                    exp_no += 1
                    results_df.to_pickle(checkpoint_path)
                    if fm == "FRR2": frr2_flag=True
    results_df.to_pickle("out/results/exps_{}_clustering.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
    
if __name__ == "__main__":
    main()