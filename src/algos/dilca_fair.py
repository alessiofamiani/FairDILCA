from algos.dilca import Dilca
import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd

class DilcaFair(Dilca):
    
    def __init__(self, sensitive, alpha = 0.5, fair_method = 'FM', k = 3, method = 'M', sigma = 1, distance_fairness=True, missing_values=None, mv_handling='mean_mode', dtypes='infer', discretize='kmeans', n_bins = 5, dp = False, preprocessing = True):
        """
        Create a DilcaFair model and sets the required parameters.
        
        :param alpha: hyperparameter between 0 (not fair) and 1 (fair).
        :param sensitive: index of the protected feature in the given dataset.
        :param fair_method: the type of fair context selection method, either Fair M (FM) or Fair RR (FRRX) where X is the number of the variant for FRR.
        :param distance_fairness: flag that indicates whenever use the fair distance computation with alpha.
        """
        Dilca.__init__(self, method, sigma, k, missing_values, mv_handling, dtypes, discretize, n_bins, dp, preprocessing)
        if alpha >= 0.0 and alpha <= 1.0: self.alpha = alpha
        else: raise ValueError("Alpha parameter must be between 0 and 1.")
        self.sensitive = sensitive
        self.fair_method = fair_method
        self.distance_fairness = distance_fairness

        self._fair_fitted = False
        
    def recompute_sensitive(self, X, sensitive):
        """
        Recompute sensitive attribute index when discretize=drop.

        Args:
            X (array-like): data.
            sensitive (int): the index of the protected feature in the given dataset.

        Returns:
            int: the updated sensitive attribute index.
        """
        new_sensitive = sensitive
        if self.discretize == 'drop':
            cols = list(X.columns.values)
            if is_numeric_dtype(cols[sensitive]): raise ValueError("Sensitive attribute drop.")
            for c in cols: 
                # check if column contains numeric data
                if is_numeric_dtype(X[c]) and c < sensitive: 
                    new_sensitive = new_sensitive - 1
        return new_sensitive

    def fit_fair(self, X, y=None, verbose=False):
        """
        Fit data X into the model.
        Args:
            X (array-like): data.

        Raises:
            ValueError: Method should be in {FM, FRR1, FRR2}
        """
        if self.sensitive not in range(len(X.columns.values)): raise ValueError("Sensitive attribute index must be in [0-{}].".format(len(X.columns.values)-1))
        self.sensitive = self.recompute_sensitive(X, self.sensitive)
        # fit non fair version, compute SU matrix etc...
        if not self._fitted: self.fit(X, y=y)
        # if discretize==drop, we need to re-compute the sensitive attribute index
        if verbose: 
            if self.discretize == 'drop':
                print("number of features after drop: ", self._m)
                print("new sensitive attribute index: ", self.sensitive)
        
        # indexes of non sensitive attributes
        self._ns_attributes_indexes = [a for a in range(0, self._m) if a != self.sensitive]
  
        self._context_fair = []
        self._distance_list_fair = []
        self._fsu_matrix = self._compute_fsu(self.alpha, self.sensitive, self._su_matrix)

        # apply fair context selection method
        for t in range(self._m):
            if self.fair_method == 'FM':
                self._context_fair.append(self._context_M_fair(t, self.sigma))
            elif self.fair_method == "FRR1" or self.fair_method == "FRR2":
                self._context_fair.append(self._context_RR_fair(t))
            elif self.fair_method == "B":
                self._context_fair.append(self._context_baseline(t))
            else: raise ValueError("Method must be in {'B', 'FM', 'FRR1', 'FRR2'}")
            if self._context_fair[t] == [] and t != self.sensitive: raise Exception("Cannot return a context.")
            # compute distance accross attribute values
            self._distance_list_fair.append(self._distance_fair(t, self.sensitive, self._su_matrix, c = self._context_fair))
        self._fair_fitted = True

    def _compute_fsu(self, alpha, sensitive, su_matrix):
        """
        Compute Fair Symmetric Uncertainty.

        Args:
            alpha (float): parameter that controls the amount of fairness.
            sensitive (int): index of the protected feature in the given dataset.
            su_matrix (array-like): symmetric uncertainty values for every couple of attributes.

        Returns:
            array-like: fair symmetric uncertainty values for every couple of attributes.
        """
        fsu_matrix = np.zeros((self._m, self._m)) 
        #su_matrix = (fsu_matrix - alpha)*su_matrix + (alpha*(1.0-su_matrix[:, sensitive]))
        #BUG: perché da valori diversi da quello sopra? operatore @?
        for i in range(0, self._m):
            for j in range(0, self._m):
                fsu_matrix[i,j] = (1-alpha)*su_matrix[i,j] + alpha*(1-su_matrix[i, sensitive])
        return fsu_matrix
    
    # FIXME: da rivedere o levare
    def proxies(self, sensitive):
        # add attributes indexes column
        a = np.c_[self._fsu_matrix, range(self._m)]
        # decreasing ranking of FSUs
        a_sorted = a[a[:, sensitive].argsort()][::-1]
        # attribute with the highest FSUs
        context = a_sorted[:,-1]
        return [int(e) for e in context]
    
    def _context_baseline(self, target):
        """
        Compute the context by simply using the (non-fair) context selection method given and removing the sensitive attribute        
        Args:
            target (int): the target attribute index.

        """
        if target == self.sensitive: return []
        if self.method == 'M': context = self._context_M(target, self.sigma, self._su_matrix)
        elif self.method == 'RR': context = self._context_RR(target)
        else: raise ValueError("method must be in {'M','RR'}")
        if self.sensitive in context: context.remove(self.sensitive)
        return context
    
    def _context_M_fair(self, target, sigma):
        """
        Compute the context using the mean of the v values.
        :param target: the target attribute index.
        :param sigma: parameter that controls the influence of the mean value.
        """
        # exclude sensitive attribute from the context selection
        if target == self.sensitive: return []
        # exclude both target and sensitive attributes from the mean computation
        indexes = list(set([i for i in range(0, self._m)]) - set([target, self.sensitive]))
        v_vector = self._fsu_matrix[:, target]
        mean = np.sum(v_vector[indexes])/ (self._m - 2)
        # collect attributes which have FSU \geq sigma*mean (and exclude both target and sensitive ones)
        context = [i for i in range(self._m) if v_vector[i] >= sigma * mean]
        return [c for c in context if c != target and c != self.sensitive]

    def _context_RR_fair(self, target):
        """
          Compute the context using FCBF.
        - FRR1 is similar to the non fair method but uses FSU instead of SU;
        - FRR2 uses the non fair method to obtain the context for a target attribute and then it removes attributes from it  if \alpha * SU_s[i] > (1-\alpha)*SU_y[i]

        Args:
            target (int): the target attribute index.
        """
        if target == self.sensitive: return []
        
        context = []
        if self.fair_method == 'FRR1':
            # add attributes indexes column
            a = np.c_[self._fsu_matrix, range(self._m)]
            # decreasing ranking of FSUs
            a_sorted = a[a[:, target].argsort()][::-1]
            # attribute with the highest FSUs
            context = [int(a_sorted[0,-1])]
            # from 1 in order to remove target 
            for i in range(1, self._m -1):
                # if attribute is not redundant and not the sensitive one then add it to the context
                if a_sorted[i, target] > np.max([a_sorted[i,c] for c in context]):
                    if int(a_sorted[i, -1]) != self.sensitive:
                        context.append(int(a_sorted[i, -1]))
        elif self.fair_method == 'FRR2':
            """
            context = self._context_RR(target)
            SU_s = self._su_matrix[:, self.sensitive]
            SU_y = self._su_matrix[:, target]
            for i in context.copy():
                if (self.alpha * SU_s[i] > (1-self.alpha)*SU_y[i]) or (i == self.sensitive):
                    context.remove(i)
            """
            # -----
            SU_s = self._su_matrix[:, self.sensitive]
            a = np.c_[self._su_matrix, range(self._m)]
            a_sorted = a[a[:,target].argsort()][::-1]
            context = [int(a_sorted[0,-1])]
            for i in range(1,self._m -1):
                # qui si controlla che non sia ridondante e che non agisca come proxy
                #if (a_sorted[i,target] > np.max([a_sorted[i,c] for c in context])) and ((1-self.alpha)*a_sorted[i, target] > self.alpha*SU_s[i]):
                if (a_sorted[i,target] > np.max([a_sorted[i,c] for c in context])) and (a_sorted[i, target] > SU_s[i]):
                    context.append(int(a_sorted[i,-1])) 
            # ------
        if self.sensitive in context: context.remove(self.sensitive) #FIXME: trovare altro modo
        context.sort()
        return context

    def _compute_alphas(self, sensitive, su_matrix, n):
        """
        Compute alpha_i values.
        :param sensitive: the protected attribute index.
        :param su_matrix: symmetric uncertainty values for every couple of attributes.
        :param n: length of the context of the target attribute.
        """
        SU_SX = su_matrix[:, sensitive]
        alphas = np.ones(self._m)
        alphas = (alphas - SU_SX) / (n - np.sum(SU_SX))
        return alphas

    def _distance_fair(self, target, sensitive, su_matrix, c, T_list=0):
        """Compute distances between target attribute values.

        Args:
            target (int): the target attribute index.
            sensitive (int): the protected attribute index.
            su_matrix (array-like): symmetric uncertainty values for every couple of attributes.
            c (list): a list containing the context for each attribute.
            T_list (array-like): list of contingency tables. Defaults to 0.

        Returns:
            array-like: a matrix containing distances between target attributes values.
        """
        # exclude sensitive attribute from the computation
        if target == sensitive: return None
        # the context of the target attribute
        context = c[target]
        if self.distance_fairness:
            alphas = self._compute_alphas(sensitive, su_matrix, len(context))
        # the unique values of the target attribute
        k = len(set(self._dataset.iloc[:,target]))
        # probability tensor, k x k x len(context) where the first two are indexed and the cell is a list of len(context) position
        P_tensor = np.zeros((k,k,len(context)))
        n = np.zeros(len(context))
        # h is the index and i is the attribute (index)
        for h,i in enumerate(context):
            if T_list == 0:
                # i guess this is for symmetrical purposes of the matrix
                if i < target:
                    T = self._contingency_tables[i][target].transpose()
                else:
                    T = self._contingency_tables[target][i]
            else:
                print("T_list[i]")
                # contingency table between target and variable i
                T = T_list[i]
            # number of column of the contingency table, cardinality of an attribute i belonging to the context of target 
            n[h] = T.shape[1]
            # should be probability of Y, P(Y)
            Ty = np.sum(T, axis = 0)
            # guess for avoiding divisions by zerlo
            Ty[Ty==0]=1
            # conditional probabilities
            P = np.array(T/Ty)
            # should be y_i
            for a in range(k):
                # should be y_j
                for b in range(a+1,k):
                    # h should be x_k
                    if self.distance_fairness:
                        P_tensor[a,b,h] = P_tensor[b,a,h] = np.sum(alphas[i] * np.power(P[a,:]-P[b,:],2))
                    else:
                        P_tensor[a,b,h] = P_tensor[b,a,h] =np.sum(np.power(P[a,:]-P[b,:],2))
        #if self._prova:
        #    print(P_tensor)
        if self.distance_fairness:
            DistanceM = np.sqrt(np.sum(P_tensor, axis = 2)/np.sum(n*alphas[i]))
        else:
            DistanceM = np.sqrt(np.sum(P_tensor, axis = 2)/np.sum(n))

        return DistanceM

    def encoding_fair(self, dataset = [], distance_list = []):
            if len(distance_list) == 0:
                distance_list = self._distance_list
            if len(dataset) == 0:
                dataset = self._dataset

            if dataset.shape[1] != self._m:
                raise ValueError(f'Wrong data shape: dataset must have {self._m} columns') 

            # compute new coordinates for each feature
            X_list = []
            for k in range(self._m):
                if k == self.sensitive: X_list.append([])
                else:
                    D = distance_list[k]
                    n = D.shape[0]
                    M = np.zeros((n,n))
                    for i in range(n):
                        for j in range(n):
                            M[i,j] = 1/2*(D[0,i]**2 + D[0,j]**2 - D[i,j]**2)
                    # eigen values
                    S=np.linalg.eig(M)[0]
                    S[abs(S)<.00001] = 0
                    # eigen values
                    U = np.linalg.eig(M)[1]
                    # matrix product @
                    X = U@np.sqrt(np.diag(S))
                    X = X[:,np.sum(X, axis = 0) != 0]
                    X_list.append(np.real(X))
            new = pd.DataFrame()
            for f in self._ns_attributes_indexes:
                x_shape = X_list[f].shape[1]
                for k in range(x_shape):
                    new[f'{f}_{k}'] = dataset.apply(lambda x: X_list[f][x.iloc[f],k], axis = 1)
            return new