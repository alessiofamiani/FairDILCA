
from collections import Counter
import numpy as np

# FAIRNESS METRICS

# CLUSTERING
def create_clusters(sens_column, labels):
    clusters = {}
    for i, l  in enumerate(labels):
        if l not in clusters: clusters[l] = []
        clusters[l].append(sens_column[i])
    return clusters

def max_fairness_cost(dataset, sensitive, labels):
    groups = dataset[sensitive].unique()
    indices = dataset[sensitive]
    
    clusters = create_clusters(indices, labels)
    
    n = dataset.shape[0]
    ideal_props = [((Counter(indices)[g])/n) for g in groups]
    
    MFC_C = []
    for c in clusters.keys():
        MFC_C_G = []
        for i, g in enumerate(groups):        
            count_g = Counter(clusters[c])[g]
            P_g = count_g/(len(clusters[c]))
            d = abs(ideal_props[i]- P_g)
            MFC_C_G.append(d)
        MFC_C.append(MFC_C_G)
    return MFC_C

def entropy(value):
    h = - value * np.log(value + 1e-5)
    return h

def balance_entropy(dataset, sensitive, labels):
    groups = dataset[sensitive].unique()
    d_s = dataset[sensitive]
    clusters = create_clusters(d_s, labels)
    
    entropy_groups = []
    for g in groups:
        entropy_clusters = []
        for c in clusters.keys():
            r_i_f = Counter(clusters[c])[g]/len(clusters[c])
            h = entropy(r_i_f)
            entropy_clusters.append(h)
        entropy_groups.append(sum(entropy_clusters))
    return entropy_groups

def balance_gen(dataset, sensitive, labels, delta=0.2):
    groups = dataset[sensitive].unique()
    indices = dataset[sensitive]
    
    clusters = create_clusters(indices, labels)
    
    #fairness_clusters = []
    balance_clusters = []
    for c in clusters.keys():
        balance_groups = []
        #fairness_groups = []
        for g in groups:
            r_i = Counter(indices)[g]/dataset.shape[0]
            r_i_f = Counter(clusters[c])[g]/len(clusters[c])
            if r_i != 0 and r_i_f !=0: balance_c = min(r_i/r_i_f, r_i_f/r_i)
            else: balance_c = 0.0
            balance_groups.append(balance_c)
            #beta_i = r_i*(1-delta) #lb
            #alpha_i = r_i/(1-delta)
            #print(beta_i, alpha_i)
            
        balance_clusters.append(balance_groups)
    return balance_clusters     

# CLASSIFICATION
# FIXME: check DivisionByZero
# TODO: extends to work with misclassification costs
# TODO: extends to work with multiple pos_label
def selection_rate(y_pred, pos_label, group_membership=None, group_label=None):
    pos = y_pred[y_pred == pos_label]
    n_pos = len(pos)
    n_samples = len(y_pred)
    if group_membership is not None:
        if group_label != None:
            zipped = [el for el in zip(y_pred, group_membership)]
            n_yg_and_ypos = sum([1 for y,s in zipped if y == pos_label and s == group_label])
            selection_rate = n_yg_and_ypos/n_samples
        else: raise ValueError("Cannot compute selection rate og group, group_label missing.")
    else: selection_rate = n_pos/n_samples
    return selection_rate

def demographic_parity_ratio(y_pred, pos_label, group_membership):
    selection_rates = []
    for g in group_membership.unique().tolist():
        selection_rates.append(selection_rate(y_pred, pos_label, group_membership=group_membership, group_label=g ))
    return min(selection_rates)/max(selection_rates)

def demographic_parity_difference(y_pred, pos_label, group_membership):
    selection_rates = []
    for g in group_membership.unique().tolist():
        selection_rates.append(selection_rate(y_pred, pos_label, group_membership=group_membership, group_label=g ))
    return abs(max(selection_rates)-min(selection_rates))

def equal_opportunity_difference(y_true, y_pred, pos_label, group_membership):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    recalls = []
    # for all groups
    for g in group_membership.unique():
        # filter by group
        indices = np.where(group_membership == g)
        #recalls.append(recall_score(y_true_array[indices], y_pred_array[indices], pos_label=pos_label, average="macro"))
        recalls.append(true_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
    return abs(max(recalls)-min(recalls))

def equal_opportunity_ratio(y_true, y_pred, pos_label, group_membership):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    recalls = []
    # for all groups
    for g in group_membership.unique():
        # filter by group
        indices = np.where(group_membership == g)
        #recalls.append(recall_score(y_true_array[indices], y_pred_array[indices], pos_label=pos_label, average="macro"))
        recalls.append(true_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
    return min(recalls)/max(recalls)

def true_positive_rate(y_true, y_pred, pos_label):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # labels identified with pos_label that are pos_label
    zipped = [el for el in zip(y_true_array, y_pred_array)]
    n_pos = sum([1 for true, pred in zipped if true==pos_label])
    if n_pos == 0.0: return 0.0
    return sum([1 for true, pred in zipped if pred==pos_label and true==pos_label])/n_pos

def false_positive_rate(y_true, y_pred, pos_label):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # labels identified with pos_label that aren't pos_label
    zipped = [el for el in zip(y_true_array, y_pred_array)]
    n_neg = sum([1 for true, pred in zipped if true!=pos_label])
    if n_neg == 0.0: return 0.0
    return sum([1 for true, pred in zipped if pred==pos_label and true!=pos_label])/n_neg 

def equalized_odds_difference(y_true, y_pred, pos_label, group_membership):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    tprs = []
    fprs = []
    
    for g in group_membership.unique():
        indices = np.where(group_membership == g)
        #tprs.append(recall_score(y_true_array[indices], y_pred_array[indices], pos_label=pos_label, average="macro"))
        tprs.append(true_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
        fprs.append(false_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
    tpr_diff = abs(max(tprs)-min(tprs))
    fpr_diff = abs(max(fprs)-min(fprs))
    return max(tpr_diff, fpr_diff) 

def equalized_odds_ratio(y_true, y_pred, pos_label, group_membership):
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    tprs = []
    fprs = []
    
    for g in group_membership.unique():
        indices = np.where(group_membership == g)
        #tprs.append(recall_score(y_true_array[indices], y_pred_array[indices], pos_label=pos_label, average="macro"))
        tprs.append(true_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
        fprs.append(false_positive_rate(y_true_array[indices], y_pred_array[indices], pos_label=pos_label))
    tpr_ratio = min(tprs)/max(tprs)
    fpr_ratio = min(fprs)/max(fprs)
    return min(tpr_ratio, fpr_ratio)
 
# METRICS
def intersection(list_1, list_2):
    return [el for el in list_1 if el in list_2]

def recall(context, fair_context):
    return len(intersection(context, fair_context))/ len(context)

def precision(context, fair_context):
    return len(intersection(context, fair_context))/ len(fair_context)

def f_score(context, fair_context):
    try:
        return 2 * ((precision(context, fair_context) * recall(context, fair_context))/ (precision(context, fair_context) + recall(context, fair_context)))
    except ZeroDivisionError:
        return 0.0

def central_point_index(matrix):
    # Note: group objs as rows
    sum = matrix.sum(axis=1)
    return np.argmin(sum)
 
def bcss(dataset, sensitive, matrix):
    dataset_center = matrix[central_point_index(matrix)]
    groups = dataset[sensitive].unique()
    
    bcss = 0.0
    for g in groups:
        group_indexes = np.where(dataset[sensitive] == g)[0]
        group_center_index = central_point_index(matrix[group_indexes])
        group_center = matrix[group_center_index]
    
        distance = np.linalg.norm(dataset_center - group_center, ord=2)
        bcss += (len(group_indexes)/dataset.shape[0]) * (distance**2)
    return bcss

def compute_groups_means(dataset, sensitive, objs_matrix, normalise=True):
    # Normalise matrix
    if normalise: normalised_matrix = objs_matrix/objs_matrix.max()
    else: normalised_matrix = objs_matrix
    # Obtaining groups indices and sizes
    groups = dataset[sensitive].unique()
    groups_indices = [np.where(dataset[sensitive] == g)[0] for g in groups]
    group_sizes = [gi.shape[0] for gi in groups_indices]
    # State the majority group (the most represented one)
    majority_group_index = np.argmax(group_sizes)
    majority_group = groups[majority_group_index]
    # Computing MaG-MiG indices 
    means = np.zeros(len(groups))
    for i in range(len(groups)):
        MaG_MiG_indices = np.ix_(groups_indices[majority_group_index], groups_indices[i])
        # computing mean
        mag_mig_matrix = normalised_matrix[MaG_MiG_indices]
        means[i] = mag_mig_matrix.mean()
    # Computing MaG-Others (MiGs + MaG)
    MaG_MiGs_indices = np.ix_(groups_indices[majority_group_index], np.arange(normalised_matrix.shape[1]))
    means_MaG_Migs = normalised_matrix[MaG_MiGs_indices].mean()
    
    return means, means_MaG_Migs, groups, group_sizes, majority_group