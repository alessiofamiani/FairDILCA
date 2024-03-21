from algos.dilca_fair import DilcaFair
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# TESTS
def check_context_methods(dilca_fair, dilca, sensitive, methods):
    df = dilca_fair
    d = dilca
    # if alpha equals zero
    for index, pair in enumerate(zip(df, d)):
        if index != sensitive:
            if pair[0] != pair[1]:
                if sensitive not in pair[1]:
                    print("{} error: {}".format(methods, pair))
                    return False
        else: 
            if pair[0] != []:
                print("{} error: {}".format(methods, pair))
                return False 
    return True

def check_context_alpha0 (X, sensitive, y=None, discretize='kmeans', mv_handling="mean_mode", missing_values=None):
    statuses = []
    checks = [('FM', 'M'), ('FRR1', 'RR'), ('FRR2', 'RR')]
    for fm, m in checks:
        model = DilcaFair(discretize=discretize, sensitive=sensitive, alpha=0.0, sigma=1.0, fair_method=fm, method=m, mv_handling=mv_handling, missing_values=missing_values)
        model.fit_fair(X, y=y)
        new_s = model.recompute_sensitive(X, sensitive)
        status = check_context_methods(model._context_fair, model._context, new_s, (fm, m))
        statuses.append(status)
        #if not status: raise ValueError("Dissimilar contexts for methods [{}-{}]for alpha=0.".format(fm, m))
    return statuses
  
def check_s_in_context(model):
    sensitive = model.sensitive
    fair_context = [e for l in model._context_fair for e in l]
    if sensitive in fair_context: raise ValueError("Sensitive attribute {} present in contexts".format(sensitive))