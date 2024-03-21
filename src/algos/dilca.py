import logging
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import binom
from itertools import combinations

class Dilca(BaseEstimator):
    """Dilca algorithm (Ienco, 2012).

     ------------------------
    |                        |
    |It requires pandas v1.1 |
    |                        |
     ------------------------

    Parameters
    ----------

    Attributes
    ----------


    """

    def __init__(self, method = 'M', sigma = .5, k = 3, missing_values=None, mv_handling='delete_rows', dtypes='infer', discretize='kmeans', n_bins = 5, dp = False, preprocessing = True):
        """
        Create the model and initialize the required parameters.

        :type method: str
        :param method: the context selection strategy, one of {'M','RR'}. Default: 'M'
        :type sigma: float
        :param sigma: the sigma parameter, only needed if method = 'M'.
        :param missing_values: missing values format, if it differs from None
        :param mv_handling: how to handle missing data, one of {'delete_rows','mean_mode','random'}
        :param dtypes: how to set column types, one of {'infer', 'keep'}
        :param discretize: How to handle numerical features, one of {'drop', 'uniform','quantile','kmeans'}
                        * drop: drop non categorical features
                        * uniform, quantile and kmeans require the specification of the number of bins for each numerical feature
        :type n_bins: int or None
        :param n_bins: Only needed if discretize != 'drop'. If None, n_bins is set equal to the square of the number of records (rounded up).
                        The same n_bins will be applied to all numerical features.
        :param preprocessing: whether to preprocess the data               

        """

        self.method = method
        self.sigma = sigma
        self.missing_values = missing_values
        self.mv_handling = mv_handling
        self.dtypes = dtypes
        self.discretize = discretize
        self.n_bins = n_bins
        self._fitted = False
        self.dp = dp
        self.k = k
        self.preprocessing = preprocessing

        np.seterr(all='ignore')


    def _init_all(self, X, y=None):
        # FIXME: non funziona con delete_rows
        # TODO: recompute sensitive quando si droppa una colonna con un solo valore
        """
        Prepare the dataset and initialize all variables.

        :param df: the dataset (a pandas.DataFrame)
        :return:
        """

        # dataset pre-processing
        self._have_labels = False
        self._y = pd.DataFrame(y, copy=True)
        if not self._y.isnull().values.any():
            if len(self._y) == X.shape[0]:
                self._have_labels = True
        self._dataset = pd.DataFrame(X, copy=True)
        self._m = self._dataset.shape[1]
            

        self._dataset.columns = range(self._m)

        if self.preprocessing:
            #self._col_dictionary= {i:i for i in self._m}
            #self._dataset.where(self._dataset.notnull(), None, inplace=True)
            if self.missing_values != None:
                self._dataset.replace(self.missing_values,np.nan,inplace=True)

            if self.dtypes == 'infer':
                self._dataset = self._dataset.infer_objects()

            if self.mv_handling=='mean_mode':
                #print('a')
                d_num = self._dataset.select_dtypes(include='number')
                d_num.fillna(d_num.mean(), inplace=True)
                d_obj = self._dataset.select_dtypes(exclude='number')
                if not d_obj.mode().empty: #TODO: check 
                    d_obj.fillna(d_obj.mode().iloc[0,:], inplace=True) # FIXME if not exists
                self._dataset = pd.concat([d_num,d_obj],axis=1)[range(self._m)]
                            
            elif self.mv_handling=='delete_rows':
                #print('b')
                if self._have_labels:
                    a = pd.concat([self._dataset,self._y], axis=1).dropna()
                    a.reset_index(inplace=True)
                    self._dataset = a.iloc[:, :-1]
                    self._y = a.iloc[:, -1]
                else:
                    self._dataset.dropna(inplace=True)
                    self._dataset.reset_index(inplace = True)

            
            d_num = self._dataset.select_dtypes(include='number')
            d_obj = self._dataset.select_dtypes(exclude='number')

            d_le = pd.DataFrame()
            le = LabelEncoder()
            for i,j in enumerate(d_obj.columns):
                le.fit(list(set(d_obj.iloc[:,i])))
                d_le[j] = le.transform(d_obj.iloc[:,i])
                

            self._n = self._dataset.shape[0]
            if not d_num.empty:
            
                if self.discretize == 'drop': #cancella colonne numeriche (se deve).
                    #print('c')
                    #self._dataset = self._dataset.select_dtypes(exclude='number')
                    self._dataset = d_le
                    self._m = self._dataset.shape[1]
                    self._dataset.columns = range(self._m)
            
            
                else: #discretizzazione colonne numeriche.
                    #print('d')
                    if self.n_bins == None:
                        self.n_bins = int(np.max((np.ceil(np.sqrt(self._n)),2)))
                    discretizer = KBinsDiscretizer(n_bins = self.n_bins, encode = 'ordinal', strategy = self.discretize)
                    d_num_header = d_num.columns
                    d_num = pd.DataFrame(discretizer.fit_transform(d_num), columns = d_num_header, dtype = np.int16)          
                    self._dataset = pd.concat([d_num,d_le],axis=1)[range(self._m)]
            else:
                self._dataset = d_le

            
            l = [i for i in range(self._m) if len(set(self._dataset.iloc[:,i].values)) == 1]
            self._dataset.drop(l, axis = 1, inplace=True)
            self._m = self._dataset.shape[1]
            self._dataset.columns = range(self._m)
                               
        else:
            self._n = self._dataset.shape[0]
        

        # variable initialization
        self._contingency_tables = []                   # list of all contingency tables between pairs of variables
        self._entropies = np.zeros(self._m)             # entropy of each feature
        self._joint_entr_matrix = np.zeros((self._m, self._m)) # entropy of each pair of features
        self._su_matrix = np.zeros((self._m, self._m))  # matrix of the su
        self._mi_matrix = np.zeros((self._m, self._m))  # matrix of the mi
        self._context = []                              # list of contexts
        self._distance_list = []                        # list of distances between values
        #self._obj_distance_matrix = np.zeros((self._n, self._n)) # distances between rows
        self._k = np.zeros(self._m, dtype = int)


    def fit(self, X, y=None):
        """
        Fit Dilca to the provided data.

        Parameters
        ----------

        X: array-like, the input dataset
        y: labels
        """

        self._init_all(X, y)
        
        for i in range(self._m):
            self._contingency_tables.append(self._create_contingency_matrices(i))
            if i == 0:
                self._k[0] = self._contingency_tables[0][1].shape[0]
                for j in range(1,self._m):
                    self._k[j] = self._contingency_tables[0][j].shape[1]

        for i in range(self._m):
            self._entropies[i] = self._compute_entropy(i)
            for j in range(i+1, self._m):
                self._joint_entr_matrix[i,j]=self._joint_entr_matrix[j,i]=self._compute_joint_entropy(i,j)
                #print(i,j,self._joint_entr_matrix[i,j], self._compute_joint_entropy(i,j))
                    
        for i in range(self._m):
            for j in range(i+1, self._m):
                self._mi_matrix[i,j]=self._compute_mi(i,j)
                self._mi_matrix[j,i]=self._compute_mi(j,i)
                self._su_matrix[i,j] = self._su_matrix[j,i] = self._compute_su(i,j)

        if self.dp == False:

            for i in range(self._m):
                if self.method == 'RR':
                    self._context.append(self._context_RR(i))
                elif self.method == 'M':
                    self._context.append(self._context_M(i, self.sigma, self._su_matrix))
                elif self.method == 'K':
                    self._context.append(self._context_K(i, self.k))
                elif self.method == 'K_dep':
                    self._context.append(self._context_K_dep(i, self.k))
                else:
                    raise ValueError("method must be in {'M','RR','K', 'K_dep'}") 
                self._distance_list.append(self._distance(i, c = self._context))

        self._fitted = True


    def distance_matrix(self, distance_list = []):
        if len(distance_list) == 0:
            distance_list = self._distance_list
            
        self._obj_distance_matrix = np.zeros((self._n, self._n)) # distances between rows
        
        d = np.array(self._dataset)
        
        for i in range(self._n):
            for j in range(i+1, self._n):
                self._obj_distance_matrix[i,j]= self._obj_distance_matrix[j,i]= np.sqrt(np.sum([np.power(distance_list[z][d[i,z],d[j,z]],2) for z in range(self._m)]))

    def encoding(self, dataset = [], distance_list = []):
        if len(distance_list) == 0:
            distance_list = self._distance_list
        if len(dataset) == 0:
            dataset = self._dataset

        if dataset.shape[1] != self._m:
            raise ValueError(f'Wrong data shape: dataset must have {self._m} columns') 

        # compute new coordinates for each feature
        X_list = []
        for k in range(self._m):
            D = distance_list[k]
            n = D.shape[0]
            M = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    M[i,j] = 1/2*(D[0,i]**2 + D[0,j]**2 - D[i,j]**2)
            S=np.linalg.eig(M)[0]
            S[abs(S)<.00001] = 0
            U = np.linalg.eig(M)[1]
            X = U@np.sqrt(np.diag(S))
            X = X[:,np.sum(X, axis = 0) != 0]
            X_list.append(np.real(X))
        new = pd.DataFrame()
        for f in range(self._m):
            for k in range(X_list[f].shape[1]):
                new[f'{f}_{k}'] = dataset.apply(lambda x: X_list[f][x.iloc[f],k], axis = 1)
        return new

    def check_dataset(self, dataset, train_dataset = []):
        if len(train_dataset) == 0:
            train_dataset = self._dataset
        for i in range(self._m):
            s = set(dataset[i])
            s_train = set(train_dataset[i])
            if not s.issubset(s_train):
                #raise ValueError(f"Attribute {i} has extra values: {[x for x in s if x not in s_train]}")
                print(f"Attribute {i} has extra values: {[x for x in s if x not in s_train]}")
                return True
            
        return False
            

    def clustering(self, k, obj_dist_matrix):
        l = list()
        
        for i in range(self._n):
            for j in range(i+1, self._n):
                l.append(self._obj_distance_matrix[i,j])
        
        #model = AgglomerativeClustering(k, affinity='precomputed', linkage='average')
        #self.labels_ = model.fit_predict(self._obj_distance_matrix)

        Z = linkage(np.array(l), method = 'ward')
        self.labels_ = fcluster(Z, t=k, criterion='maxclust')
        
        self.ari_ = 0
        self.nmi_ = 0

        if self._have_labels:
            self.ari_ = ari(self.labels_, self._y)
            self.nmi_ = nmi(self.labels_, self._y)
        #else:
        #    print("Can't compute ari and nmi. Possible reasons are: y not given, len(y)!= len(labels_), y has missing values.")
        


    def _create_contingency_matrices(self, target):
        """
        Create the contingency tables between target attribute and all the other attributes that follow the target in the dataset.
        :type target: int
        :param target: the index of the target variable as column of the dataset
        :return: a list of contingency tables. Length of the list: num features. In the positions j with j <= target
                 the list contains 0 instead of a contingency table.   
        """

        t_list = []
        for i in range(self._m):
            if i > target:
                T = pd.crosstab(self._dataset.iloc[:,target],self._dataset.iloc[:,i])
            else:
                T = 0
            t_list.append(T)
        return t_list

    def _compute_entropy(self, x):
        """

        """
        if x == 0:
            T = np.array(self._contingency_tables[x][x+1].transpose())
        else:
            T = np.array(self._contingency_tables[x-1][x])

        tot = np.sum(T)
        T_prob = T/tot
        #k = len(np.shape(M))
        pX = np.sum(T_prob, axis = 0)
        out = np.zeros(pX.shape)
        hX = -np.sum(pX *np.log2(pX, out=out,where=pX>0))
        
        return hX


    def _compute_joint_entropy(self, x, y):
        """

        """
        
        if x == y:
            raise ValueError("x and y must be different.")
        elif x > y:
            T = np.array(self._contingency_tables[y][x])
        else:
            T = np.array(self._contingency_tables[x][y])

        tot = np.sum(T)
        T_prob = np.true_divide(T,tot)
        out = np.zeros(T_prob.shape)
        hXY = -np.sum(T_prob *np.log2(T_prob, out = out,where=T_prob>0))

        return hXY
        
        

    def _compute_mi(self, x, y):

        if x == y:
            su = 0
        else:
            hX = self._entropies[x]
            #hY = self._entropies[y]
            hXY = self._joint_entr_matrix[x,y]
            #su = (hX - hXY)#/(hX + hY)
            su = hX - hXY
        return su

    def _compute_su(self, x, y):

        if x == y:
            su = 0
        else:
            hX = self._entropies[x]
            hY = self._entropies[y]
            hXY = self._joint_entr_matrix[x,y]
            su = (hY + hX - hXY)/(hX + hY)
        return su

    def _contingency_tensor(self,l):
        
        T = np.zeros(self._k[l])
        a = self._dataset.groupby(l).count().iloc[:,0]
        for i in a.index:
            T[i] = a[i]
        self._T = T
        return T
            
                        
    def _context_M(self, target, sigma, su_matrix):
        su = su_matrix[:,target]  # [self._compute_su(i,target) for i in range(self._m)]
        m = np.sum(su)/(self._m - 1)
        context = [i for i in range(self._m) if su[i] >= sigma * m]
        return [c for c in context if c!= target]

    def _context_K(self, target, k):
        a = np.c_[self._mi_matrix[:,target], range(self._m)]
        a_sorted = a[a[:,0].argsort()][::-1]
        context = [int(i) for i in a_sorted[:k+1, 1]]
        context.sort()
        return [c for c in context if c!= target]

    def _context_K_dep(self, target, k):
        r = [j for j in range(self._m) if j != target]
        mi_max = -np.inf
        for i in combinations(r,k):
            #self._i = [target] + [j for j in i]
            T = self._contingency_tensor([target] + [j for j in i])
            #print(i, T.shape)
            T1 = np.sum(T, axis = 0)
            out = np.zeros(T.shape)
            mi = np.sum(T*np.log2(T/T1, out = out, where=T>0))
            del(T,T1)
            if mi > mi_max:
                mi_max = mi
                s = i
        return list(s)

    def _context_RR(self, target):
        a = np.c_[self._su_matrix, range(self._m)]
        a_sorted = a[a[:,target].argsort()][::-1]
        
        context = [int(a_sorted[0,-1])]
        for i in range(1,self._m -1):
            if a_sorted[i,target] > np.max([a_sorted[i,c] for c in context]):
                context.append(int(a_sorted[i,-1]))
        context.sort()    
            
        return context


    def _distance(self, target, c, T_list = 0):
        context = c[target]
        k = len(set(self._dataset.iloc[:,target]))
        P_tensor = np.zeros((k,k,len(context)))
        n = np.zeros(len(context))
        for h,i in enumerate(context):
            if T_list == 0:
                if i < target:
                    T = self._contingency_tables[i][target].transpose()
                else:
                    T = self._contingency_tables[target][i]
            else:
                T = T_list[i]

            n[h] = T.shape[1]
            Ty = np.sum(T, axis = 0)
            Ty[Ty==0]=1
            P = np.array(T/Ty)
            for a in range(k):
                for b in range(a+1,k):
                    P_tensor[a,b,h] = P_tensor[b,a,h] =np.sum(np.power(P[a,:]-P[b,:],2))
        #if self._prova:
        #    print(P_tensor)

        DistanceM = np.sqrt(np.sum(P_tensor, axis = 2)/np.sum(n))

        return DistanceM
                            
            
        
        

                
                
   

        
        
        

        
        

        
        
        
