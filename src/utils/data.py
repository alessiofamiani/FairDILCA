import pandas as pd
import numpy as np

def load_dataset(dataset_info, path):
    file_path = path + dataset_info['filename'] + dataset_info['ext']
    sep = dataset_info['sep']
    header = dataset_info['header']
    if header: header = "infer"
    else: header = None
    data = pd.read_table(file_path, header = header, sep = sep, skipinitialspace=True)
    #if "mv_label" in dataset_info: data.replace(dataset_info['mv_label'],np.nan,inplace=True)
    data.columns = range(data.shape[1])
    # sensitive attributes as categorical ones
    dataset_sensitives = [int(k) for k in dataset_info["sensitives"].keys()]
    for s in dataset_sensitives:
        data[s] = data[s].astype(str)
    # if target last
    if dataset_info['target'] == -1:
        data_without_target =  data.iloc[:,:dataset_info['target']]
    # if target first
    elif dataset_info['target'] == 0:
        data_without_target = data.iloc[:,1:]
    # if target middle
    else:
        data_without_target = pd.concat([data.iloc[:, :dataset_info['target']], data.iloc[:, dataset_info['target']+1:]], axis=1)
    data_without_target.columns = range(data_without_target.shape[1])
    return data, data_without_target, data.iloc[:, dataset_info['target']]

def dataset_description(data, verbose=False):
    d_num = data.select_dtypes(include='number')
    d_obj = data.select_dtypes(exclude='number')
    if verbose:
        print("No. of rows:\t\t\t{}".format(data.shape[0]))
        print("No. of features:\t\t{}".format(data.shape[1]))
        print("No. numerical features:\t\t{}".format(d_num.shape[1]))
        print("No. categorical features:\t{}".format(d_obj.shape[1]))
    return data.shape[0], data.shape[1], d_num.shape[1], d_obj.shape[1]
    
def convert_datasets(original_datasets_info, in_path, out_path):
    df = pd.DataFrame(columns=["dataset", "n_rows", "n_features", "n_numerical", "n_categorical"])
    for d in original_datasets_info.keys():
        data = load_dataset(original_datasets_info[d], in_path)
        nrows, nfeatures, n_numerical, n_categorical = dataset_description(data)
        data.to_csv(out_path + "{}.data".format(d), sep=",", header=None, index=False)
        df.loc[len(df)] = (d, nrows, nfeatures, n_numerical, n_categorical)
    df.to_csv("rsc/datasets.csv", index=False)
    return df
        
def handle_mv(X, mv=None, mode="mean_mode"):
    if mv != None: X.replace(mv,np.nan,inplace=True)
    m = X.shape[1]
    d_num = X.select_dtypes(include='number')
    d_num.fillna(d_num.mean(), inplace=True)
    d_obj = X.select_dtypes(exclude='number')
    if not d_obj.mode().empty: #TODO: check 
        d_obj.fillna(d_obj.mode().iloc[0,:], inplace=True) # FIXME if not exists
    return pd.concat([d_num,d_obj],axis=1)[range(m)]            
    