
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
from time import time
from utils.tests import check_s_in_context
from utils.metrics import bcss, f_score, compute_groups_means
from utils.data import load_dataset
from sklearn.manifold import TSNE
from algos.dilca import Dilca
from algos.dilca_fair import DilcaFair
from collections import namedtuple
from scipy.stats import pearsonr, spearmanr
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Result = namedtuple('Result', ["tsec", "p", "dl1", "f1_score",  "ps_mean", "groups", "majority_group", "means_fair", "means_MaG_MiG_fair", "means_nf", "means_MaG_MiG_fair_nf", "ec", "bcss_fair", "bcss_nf", "group_sizes", "means_nn_f", "means_MaG_Migs_nn_f", "means_nn_nf", "means_MaG_Migs_nn_nf"])

def plot_tsne(c, labels, ax):
    for g in np.unique(labels):
        i = np.where(labels == g)
        ax.scatter(c[i,0], c[i, 1], label=g)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc="upper center", framealpha=1, frameon=True, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=False, ncol=len(labels), prop={'size': 22}, markerscale=2)
    
def run_experiment_e(X, s, a, fm, m, sigma, missing_values, n_bins, obj_distances_nf, tsne, dataset, Y, dname, mode, datasets, distance_fairness=True):
    ec = False
    start_time = time()
                
    model = DilcaFair(s, alpha=a, fair_method=fm, method=m, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values, sigma=sigma, distance_fairness=distance_fairness)
    
    try:
        # Compute contexts and distances among attribute values
        model.fit_fair(X)
        check_s_in_context(model)
                    
        # Computing distances between objects pairs
        # fair setting
        new = model.encoding_fair(distance_list=model._distance_list_fair)
        obj_distances = euclidean_distances(new)
        
        if tsne != None:
            z = tsne.fit_transform(new)
            labels_sensitive = dataset[s]
            
            fig_sensitive, ax_sensitive = plt.subplots(figsize=(15, 10), dpi=300)
            fig_target, ax_target = plt.subplots(figsize=(15, 10), dpi=300)
            plot_tsne(z, labels_sensitive, ax_sensitive)
            plot_tsne(z, Y, ax_target)
            fig_sensitive.tight_layout()
            fig_target.tight_layout()
            
            if fm != "B":
                if fm == "FM":  algo_label = "M"
                elif fm == "FRR1": algo_label = "RR"
                else: algo_label = "PL"
            else:
                algo_label = "{}".format(m)
            
            fig_sensitive.savefig("out/results/{}/tsne_dilca_fair_{}_{}_s{}_a{}.png".format(dname, mode, algo_label, datasets[dname]['sensitives'][str(s)], a), bbox_inches='tight', dpi=300)
            fig_target.savefig("out/results/{}/tsne_dilca_fair_{}_{}_s{}_a{}_target.png".format(dname, mode, algo_label, datasets[dname]['sensitives'][str(s)], a), bbox_inches='tight', dpi=300)
            plt.close(fig_sensitive)
            plt.close(fig_target)
            
            del(z)
        del(new)
        # non fair settings
        #new_nf = model.encoding(distance_list=model._distance_list)
        #obj_distances_nf = euclidean_distances(new_nf)
        #del(new_nf)

        # Compute metrics
                    
        # Contexts
        f_scores = []
        for i, (c, fc) in enumerate(zip(model._context, model._context_fair)): 
            if i != model.sensitive: f_scores.append(f_score(c, fc))
        f1_score = sum(f_scores)/len(f_scores)
                    
        # Objs Matrices      
        flatten_f = obj_distances[np.triu_indices(obj_distances.shape[0], k=1)]
        flatten_nf = obj_distances_nf[np.triu_indices(obj_distances_nf.shape[0], k=1)]
                    
        p = pearsonr(flatten_f, flatten_nf)[0]
        #sp = spearmanr(flatten_f, flatten_nf)[0]
        dl1 = np.linalg.norm(flatten_f - flatten_nf, ord=1)
                    
        # Distance Matrices
        ps = []
        for i in range(model._m):
            if i == model.sensitive: continue 
            ps.append(pearsonr(model._distance_list[i].ravel(), model._distance_list_fair[i].ravel())[0])
        ps_mean = sum(ps)/len(ps)
                    
        for c in model._dataset.columns: model._dataset[c] = model._dataset[c].astype(str)

        means_nf, means_MaG_Migs_nf, groups, group_sizes, majority_group = compute_groups_means(dataset, s, obj_distances_nf)
        means_f, means_MaG_Migs_f, groups, group_sizes, majority_group = compute_groups_means(dataset, s, obj_distances)
        
        means_nn_nf, means_MaG_Migs_nn_nf, groups, group_sizes, majority_group = compute_groups_means(dataset, s, obj_distances_nf, normalise=False)
        means_nn_f, means_MaG_Migs_nn_f, groups, group_sizes, majority_group = compute_groups_means(dataset, s, obj_distances, normalise=False)
        
        bcss_fair = bcss(model._dataset, model.sensitive, obj_distances/obj_distances.max())
        bcss_nf = bcss(model._dataset, model.sensitive, obj_distances_nf/obj_distances_nf.max())
          
    except Exception as e:
        if str(e) == "Cannot return a context.":
            print("Empty context")
            ec = True
            p = dl1 = f1_score = ps_mean = majority_group = bcss_fair = bcss_nf = None
            means_fair = means_nf = []
            groups = []
        else: raise ValueError(e)
    end_time = time()
    tsec = end_time-start_time

    return Result(tsec=tsec, p=p, f1_score=f1_score, dl1=dl1, ps_mean=ps_mean, groups=groups, majority_group=str(majority_group), means_fair=means_f, means_nf=means_nf, bcss_fair=bcss_fair, bcss_nf=bcss_nf, ec=ec, means_MaG_MiG_fair=means_MaG_Migs_f, means_MaG_MiG_fair_nf=means_MaG_Migs_nf, group_sizes=group_sizes, means_nn_nf=means_nn_nf, means_MaG_Migs_nn_nf=means_MaG_Migs_nn_nf, means_nn_f=means_nn_f, means_MaG_Migs_nn_f=means_MaG_Migs_nn_f)

def main():
    parser = argparse.ArgumentParser(description="FairDILCA experiments")
    parser.add_argument("dataset_name", type=str, help="Name of one of the available datasets in the repo")
    parser.add_argument("n_bins", type=int, help="Number of bins to be used by discretization technique")
    parser.add_argument("-ms", "--max_size", type=int, help="Limit row numbers for the specified dataset", default=None)
    parser.add_argument("-tsne", "--tsne_flag", choices=['True', 'False'], help="Flag for tsne computation", default=False)
    
    args = parser.parse_args()

    dname = args.dataset_name
    max_size = args.max_size # max number of records per dataset
    n_bins = args.n_bins
    tsne_flag = bool(args.tsne_flag)
    
    if not os.path.exists("out/results"): os.mkdir("out/results")

    # load datasets info
    f = open("rsc/datasets.json")
    datasets = json.load(f)
    f.close()

    datasets_path = "rsc/originals/"
    seed = 0
    
    st = datasets[dname]
    datasets = {
        dname: st
    }

    # settings
    contexts = [('M', 'FM'), ('RR', 'FRR1'), ('RR', 'FRR2')]
    fdd_contexts = [('M', 'B'), ('RR', 'B')]
    alphas = [round(a, 2) for a in np.arange(0.0, 1.0, 0.05)]
    
    cols = ['dataset', 'objs_no', 'n_bins', 'mode', 'alpha', 'sensitive', 'fair_method', 'method', 'empty_context', 'sigma', 'pearson_objs', 'l1_dist_objs', 'pearson', 'time', 'f1', "s_groups", "majority_group", "group_sizes", "means_f", "means_MaG-MiGs_f", "means_nn_f", "means_MaG-MiGs_nn_f", "means_nn_nf", "means_MaG-MiGs_nn_nf", "means_nf", "means_MaG-MiGs_nf", "bcss_f", "bcss_nf" ]

    tsne = TSNE(n_components=2, verbose=False, random_state=seed)

    datasets_path + "{}.data".format(dname) # data
    results_d_path = "out/results/{}".format(dname) # storing results for dataset d
    checkpoint_path = "out/results/exps_{}_checkpoint.csv".format(dname) # checkpoint for dataset d
        
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

    missing_values = None
    if "mv_label" in datasets[dname]: missing_values = datasets[dname]["mv_label"]

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
        objs_distances_nf = euclidean_distances(new_nf)
        
        for s in sensitives:
            frr2_flag=False
            if tsne_flag:
                z_nf = tsne.fit_transform(new_nf)
                plt.rcParams["figure.autolayout"] = True
                #FIXME: se target di dataset è 0, s-1
                labels_sensitive = X[s]
                fig_sensitive, ax_sensitive = plt.subplots(figsize=(15, 10), dpi=300)
                fig_target, ax_target = plt.subplots(figsize=(15, 10), dpi=300)
                plot_tsne(z_nf, labels_sensitive, ax_sensitive)
                plot_tsne(z_nf, Y, ax_target)
                fig_sensitive.tight_layout()
                fig_target.tight_layout()
                fig_sensitive.savefig("out/results/{}/tsne_dilca_{}_s{}.png".format(dname, m, datasets[dname]['sensitives'][str(s)]), bbox_inches='tight', dpi=300)
                fig_target.savefig("out/results/{}/tsne_dilca_{}_target.png".format(dname, m), bbox_inches='tight', dpi=300)
                plt.close(fig_sensitive)
                plt.close(fig_target)
            else: tsne = None
            
            for a in alphas:
                if frr2_flag: continue
                if checkpoint_exists:
                    if exp_no <= from_exp:
                        if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                        if results_df['alpha'][exp_no] != a: raise ValueError("Experiment config is not the same: alpha")
                        if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                        if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                        if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                        exp_no += 1
                        continue
                sigma = 1.0
                print(dname, "# {}/{}\t (alpha: {}, fair_method: {}, sensitive: {} ({}))".format(exp_no, len(contexts)*len(sensitives)*len(alphas), a, fm, datasets[dname]['sensitives'][str(s)], s))

                results = run_experiment_e(X, s, a, fm, m, sigma, missing_values, mode='CD', n_bins=n_bins, obj_distances_nf=objs_distances_nf, tsne=tsne, dname=dname, dataset=dataset, Y=Y, datasets=datasets)

                results_df.loc[exp_no] = [dname, len(X), n_bins, 'CD', a, datasets[dname]['sensitives'][str(s)], fm, m, results.ec, sigma, results.p, results.dl1, results.ps_mean, results.tsec, results.f1_score, results.groups, results.majority_group, results.group_sizes, results.means_fair, results.means_MaG_MiG_fair, results.means_nn_f, results.means_MaG_Migs_nn_f, results.means_nn_nf, results.means_MaG_Migs_nn_nf, results.means_nf, results.means_MaG_MiG_fair_nf, results.bcss_fair, results.bcss_nf]
                exp_no += 1
                
                results_df.to_pickle(checkpoint_path)
                if fm == "FRR2": frr2_flag=True
        if tsne_flag: del(z_nf)
        del(new_nf)
    
    results_df.to_pickle("out/results/{}_exps.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

    # BASELINES
    # C
    for m, fm in contexts:
        model_nf = Dilca(method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values)
        model_nf.fit(X)
        new_nf = model_nf.encoding(distance_list=model_nf._distance_list)
        objs_distances_nf = euclidean_distances(new_nf)
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
                        if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                        exp_no += 1
                        continue
                sigma = 1.0
                print(dname, "# {}/{}\t (alpha: {}, fair_method: {}, sensitive: {} ({}))".format(exp_no, 2*len(contexts)*len(sensitives)*len(alphas), a, fm, datasets[dname]['sensitives'][str(s)], s))
                if tsne_flag == None: tsne=None
                results = run_experiment_e(X, s, a, fm, m, sigma, missing_values, n_bins=n_bins, obj_distances_nf=objs_distances_nf, tsne=tsne, dname=dname, dataset=dataset, Y=Y, distance_fairness=False, mode='C', datasets=datasets)

                results_df.loc[exp_no] = [dname, len(X), n_bins, 'C', a, datasets[dname]['sensitives'][str(s)], fm, m, results.ec, sigma, results.p, results.dl1, results.ps_mean, results.tsec, results.f1_score, results.groups, results.majority_group, results.group_sizes, results.means_fair, results.means_MaG_MiG_fair, results.means_nn_f, results.means_MaG_Migs_nn_f, results.means_nn_nf, results.means_MaG_Migs_nn_nf, results.means_nf, results.means_MaG_MiG_fair_nf, results.bcss_fair, results.bcss_nf]
                exp_no += 1
                
                results_df.to_pickle(checkpoint_path)
                if fm == "FRR2": frr2_flag=True
        del(new_nf)
    
    results_df.to_pickle("out/results/{}_exps.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)

    # D
    for m, fm in fdd_contexts:
        model_nf = Dilca(method=m, sigma=1.0, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values)
        model_nf.fit(X)
        new_nf = model_nf.encoding(distance_list=model_nf._distance_list)
        objs_distances_nf = euclidean_distances(new_nf)
        for s in sensitives:
            if checkpoint_exists:
                if exp_no <= from_exp:
                    if results_df['objs_no'][exp_no] !=len(X): raise ValueError("Experiment config is not the same: objs_no")
                    if results_df['method'][exp_no] != m: raise ValueError("Experiment config is not the same: method")
                    if results_df['fair_method'][exp_no] != fm: raise ValueError("Experiment config is not the same: fair_method")
                    if results_df['sensitive'][exp_no] != datasets[dname]['sensitives'][str(s)]: raise ValueError("Experiment config is not the same: sensitive")
                    exp_no += 1
                    continue
            sigma = 1.0
            print(dname, "# {}/{}\t (alpha: {}, fair_method: {}, sensitive: {} ({}))".format(exp_no, len(sensitives)*len(fdd_contexts)+2*len(contexts)*len(sensitives)*len(alphas), a, fm, datasets[dname]['sensitives'][str(s)], s))
            if tsne_flag == None: tsne=None
            results = run_experiment_e(X, s, 0.0, fm, m, sigma, missing_values, n_bins=n_bins, obj_distances_nf=objs_distances_nf, tsne=tsne, dname=dname, dataset=dataset, Y=Y, distance_fairness=True, mode='D', datasets=datasets)
            results_df.loc[exp_no] = [dname, len(X), n_bins, 'D', a, datasets[dname]['sensitives'][str(s)], fm, m, results.ec, sigma, results.p, results.dl1, results.ps_mean, results.tsec, results.f1_score, results.groups, results.majority_group, results.group_sizes, results.means_fair, results.means_MaG_MiG_fair, results.means_nn_f, results.means_MaG_Migs_nn_f, results.means_nn_nf, results.means_MaG_Migs_nn_nf, results.means_nf, results.means_MaG_MiG_fair_nf, results.bcss_fair, results.bcss_nf]
            exp_no += 1
            
            results_df.to_pickle(checkpoint_path)
        del(new_nf)
    
    results_df.to_pickle("out/results/{}_exps.csv".format(dname))
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
            
if __name__ == "__main__":
    main()