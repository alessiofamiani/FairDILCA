import numpy as np 
from time import time
from src.algos.dilca_fair import DilcaFair
from src.utils.tests import check_s_in_context
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import pandas as pd
from collections import namedtuple    

Result = namedtuple('Result', ["tsec", "p", "sp", "dl1", "f1_score",  "ps_mean", "groups", "majority_group", "means_fair", "stds_fair", "means_nf", "stds_nf", "ec", "bcss_fair", "bcss_nf"])

# FIXME: da levare
def build_contigency_matrix(groups, cm, group_data, distance="js"):
    for i, gd1 in enumerate(group_data):
        for j, gd2 in enumerate(group_data):
            if distance == "js": cm[groups[j]][groups[i]] = jensenshannon(gd1, gd2)
            elif distance == "wd": cm[groups[j]][groups[i]] = wasserstein_distance(gd1, gd2)
            else: raise ValueError("Distance must be in {'js', 'wd'}.")
    return cm
# FIXME: da levare
def compute_divergence(groups, fair, non_fair, distance="js"):
    cont_m_nf = pd.DataFrame(index=groups, columns=groups)
    cont_m_f = pd.DataFrame(index=groups, columns=groups)

    cont_m_nf = build_contigency_matrix(groups, cont_m_nf, non_fair, distance=distance)
    cont_m_f = build_contigency_matrix(groups, cont_m_f, fair, distance=distance)
    return cont_m_f, cont_m_nf

# FIXME: da levare
def run_experiment(X, s, a, fm, m, sigma, missing_values, n_bins):
    ec = False
    start_time = time()
                
    model = DilcaFair(s, alpha=a, fair_method=fm, method=m, discretize="kmeans", mv_handling="mean_mode", n_bins=n_bins, missing_values=missing_values, sigma=sigma)
    
    try:
        # Compute contexts and distances among attribute values
        model.fit_fair(X, verbose=True)
        check_s_in_context(model)
                    
        # Computing distances between objects pairs
        # fair setting
        new = model.encoding_fair(distance_list=model._distance_list_fair)
        obj_distances = euclidean_distances(new)
        del(new)
        # non fair settings
        new_nf = model.encoding(distance_list=model._distance_list)
        obj_distances_nf = euclidean_distances(new_nf)
        del(new_nf)

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
        sp = spearmanr(flatten_f, flatten_nf)[0]
        dl1 = np.linalg.norm(flatten_f - flatten_nf, ord=1)
                    
        # Distance Matrices
        ps = []
        for i in range(model._m):
            if i == model.sensitive: continue 
            ps.append(pearsonr(model._distance_list[i].ravel(), model._distance_list_fair[i].ravel())[0])
        ps_mean = sum(ps)/len(ps)
                    
        for c in model._dataset.columns: model._dataset[c] = model._dataset[c].astype(str)
    
        # get protected groups
        groups = model._dataset[model.sensitive].unique()
        # get objs groups indexes
        groups_indexes = [np.where(model._dataset[model.sensitive] == g)[0] for g in groups]
        # get groups sizes and find the most frequent/representend group (aka the majority group)
        group_sizes = [gi.shape[0] for gi in groups_indexes]
        majority_group_index = np.argmax(group_sizes)
        majority_group = groups[majority_group_index]
        
        means_fair = np.zeros(len(groups))
        stds_fair = np.zeros(len(groups))
        means_nf = np.zeros(len(groups))
        stds_nf = np.zeros(len(groups))
        
        for i in range(len(groups)):
            MaG_MiG_indices = np.ix_(groups_indexes[majority_group_index], groups_indexes[i])
            means_fair[i] = obj_distances[MaG_MiG_indices].mean()
            stds_fair[i] = obj_distances[MaG_MiG_indices].std()
            means_nf[i] = obj_distances_nf[MaG_MiG_indices].mean()
            stds_nf[i] = obj_distances_nf[MaG_MiG_indices].std()
        
        bcss_fair = bcss(model._dataset, model.sensitive, obj_distances)
        bcss_nf = bcss(model._dataset, model.sensitive, obj_distances_nf)
          
    except Exception as e:
        if str(e) == "Cannot return a context.":
            print("Empty context")
            ec = True
            p = sp = dl1 = f1_score = ps_mean = majority_group = bcss_fair = bcss_nf = None
            means_fair = stds_fair = means_nf = stds_nf = []
            groups = []
        else: raise ValueError(e)
    end_time = time()
    tsec = end_time-start_time
    
    return Result(tsec=tsec, p=p, sp=sp, f1_score=f1_score, dl1=dl1, ps_mean=ps_mean, groups=groups, majority_group=str(majority_group), means_fair=means_fair, stds_fair = stds_fair, means_nf=means_nf, stds_nf=stds_nf, bcss_fair=bcss_fair, bcss_nf=bcss_nf, ec=ec)

# FIXME: da qui in poi spostare
def plot_group(ax, group, title, num=30):
    hist, bins = np.histogram(group.ravel(), bins=num, density=True)
    hist = hist/sum(hist)
    bins = bins/max(bins)
    ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge")
    ax.set_title(title)
    return hist

def plot_results(ax_p_sp, ax_dl1, ax_js, ax_f1, df, dataset, fair_method, sensitive):
    # retrieve the valid tuples of givens fair method and sensitive attribute
    fdf = df[(df['fair_method'] == fair_method) & (df['sensitive'] == sensitive)]
    fdf = fdf[fdf['empty_context'] == False]
    # retrieve series to plot
    alphas = fdf['alpha']
    pobjs = fdf['pearson_objs']
    sobjs = fdf['spearman_objs']
    pc = fdf['pearson']
    d1 = fdf['l1_dist_objs']
    js = fdf['js_max']
    jsf = fdf['js_max_fair']
    f1s = fdf['f1']
    
    # Plotting pearson and spearman
    ax_p_sp.plot(alphas, pobjs, label='Pearson (objects)', color='red')
    ax_p_sp.plot(alphas, sobjs, label='Spearman (objects)', color='orange')
    ax_p_sp.plot(alphas, pc, label='Pearson (distances)', color='green')
    
    ax_p_sp.set_xlabel('Alpha')
    ax_p_sp.set_ylabel('Metrics values')
    ax_p_sp.set_title("{} ({}) - {}".format(dataset, sensitive, fair_method))
    
    # Plotting L1 norm distance
    ax_dl1.plot(alphas, d1, label='Distance L1', color='blue')
    
    ax_dl1.set_ylabel('L1 Norm Distance')
    
    # Plotting JS Divergence
    ax_js.plot(alphas, js, label='Mean MaG JS Divergence (not fair)', color='purple')
    ax_js.plot(alphas, jsf, label='Mean MaG JS Divergence (fair)', color='violet')
    
    ax_js.set_xlabel('Alpha')
    ax_js.set_title("{} ({}) - {}".format(dataset, sensitive, fair_method))
    
    # Plotting F1 score
    ax_f1.plot(alphas, f1s, label='F1-Score (contexts)', color='deeppink')
    
    ax_f1.set_xlabel('Alpha')
    ax_f1.set_title("{} ({}) - {}".format(dataset, sensitive, fair_method))
    
def plot_groups_divergence(df, fair_method, sensitive, dataset):
    fdf = df[(df['fair_method'] == fair_method) & (df['sensitive'] == sensitive)]
    fdf = fdf[fdf['empty_context'] == False]
    dfl = list(zip(*fdf["div_fair_list"].to_list()))
    dnfl = list(zip(*fdf["div_nfair_list"].to_list()))
    mgs = fdf['majority_group'].to_list()[0]
    min_gs = [e for e in fdf['s_groups'].to_list()[0] if e != str(int(mgs))]
    alphas = fdf['alpha']
    fig, ax = plt.subplots(1, len(dfl), figsize=(4.5*len(dfl), 4))
    for i in range(len(dfl)):
        if len(dfl) > 1: ax_t = ax[i]
        else: ax_t = ax
        ax_t.plot(alphas, dnfl[i], label='Divergence', color='purple')
        ax_t.plot(alphas, dfl[i], label='Divergence Fair', color='violet')
        ax_t.set_xlabel('Alpha')
        #ax[i].set_ylabel('Divergence value')
        ax_t.set_title("{} MaG({}) vs MiG({})".format(dataset, str(int(mgs)), min_gs[i]))
        h1, l1 = ax_t.get_legend_handles_labels()
        fig.legend(h1, l1,loc='upper center', bbox_to_anchor=(0.5, 0.00), fancybox=True, shadow=True, ncol=5)
        fig.suptitle("{} ({}) - {}".format(dataset, sensitive, fair_method), fontweight="bold", y=1.05)