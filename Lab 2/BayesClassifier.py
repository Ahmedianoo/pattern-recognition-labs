import numpy as np

class BayesClassifier():
    def __init__(self, mode:str='QDA'):
        # the class labels (will be set upon fit)
        self.labels = [] #labels of the dataset
        # model learnable parameters will be set upon fit
        self.means = [] #means of each class
        self.covs = [] #cov matrix of each class
        self.priors = [] #each class prior
        self.weighted_cov = None        # in case of LDA only
        
        # TODO 1: Assert that mode is one of ['QDA', 'LDA', 'Naive'] and set it to self.mode
        # assert goes here
        assert mode in ['QDA', 'LDA', 'Naive'], "your mode has to be in one of ['QDA', 'LDA', 'Naive']"
        self.mode = mode

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        # TODO 2: Extract the labels from y_train using Numpy
        self.labels = np.unique(y_train)
        # TODO 3: Compute the model parameters
        for label in self.labels:
            # TODO 3.1: Extract the data belonging to the current class
            x_train_given_class = x_train[y_train == label]
            # TODO 3.2: Compute the class mean and add it to self.means
            """
                [
                [], 
                [], 
                []
                ]
            """
            class_mean = np.mean(x_train_given_class, axis=0)
            self.means.append(class_mean)
            # TODO 3.3: Compute the class covariance using np.cov and add it to self.cov
            # Check the documentation of np.cov and what layout it assumes about the input
            # Set bias=False in case of QDA only and use a one-line if for that
            # bias=False, divide by n-1, sample
            # bias=True, divide by n
            # https://numpy.org/doc/stable/reference/generated/numpy.cov.html
            # had to use rowvar=False, as the default exchanges the rows and the columns, assumes the rows are features, and cols: entries
            class_cov = np.cov(x_train_given_class, rowvar=False, bias = False if self.mode=='QDA' else True)
            self.covs.append(class_cov)
            # TODO 3.4: Compute the class prior and add it to self.priors
            class_prior = len(x_train_given_class) / len(y_train)  
            self.priors.append(class_prior)
        
        # TODO 4: Convert the model parameters from lists to numpy arrays
        self.means = np.array(self.means)
        self.covs = np.array(self.covs)
        self.priors = np.array(self.priors)
        
        # TODO 5: In case of LDA, compute the weighted covariance matrix (self.weighted_cov)
        # In one line, sum the covariance matrices for each class weighted (multiplied) by their priors
        if self.mode == 'LDA':
            self.weighted_cov = np.sum(self.covs * self.priors.reshape(len(self.labels), 1, 1), axis=0) #to sum for each class
            # reset self.covs to an empty list to save memory as we no longer need it
            self.covs = []
        # TODO 6: In case of Naive Bayes, diagonalize each of the covariance matrices to enforce independence
        elif self.mode == 'Naive':
            # diagonalize the covariance matrices in one line
            self.covs = [np.diag(np.diag(mat)) for mat in self.covs] 
            # for each matrix, take its digonal and convert it to 1D array
            # take this 1D array and construct a matrix with this array as its diagonal and zeros elsewhere
        # In case of QDA, we maintain the general setup and just return
        else: 
             return

    # TODO 7: Define the normal distribution N(x, μ, Σ) probability density function
    @staticmethod
    def N(x:np.ndarray, μ:np.ndarray, Σ:np.ndarray)->float:
        scale = np.exp((-1/2) * (x-μ).T @ np.linalg.inv(Σ) @ (x-μ))                              # start by defining the denominator 
        prob = scale / (np.power(2 * np.pi, len(x)/2) * np.sqrt(np.linalg.det(Σ)))
        return prob    
    
    # TODO 8: Given a single x, compute P(C|x) using Bayes rule for each class
    def predict_proba_x(self, x:np.ndarray)->np.ndarray:
        if self.mode != 'LDA':
            # TODO 8.1: Compute P(x|C)P(C) for each class in a numpy array in one line while using covariance matrices in self.covs
            prob_product = np.array([self.N(x, self.means[i], self.covs[i]) * self.priors[i] for i in range(len(self.labels))])
        else:
            # TODO 8.2: Compute P(x|C)P(C) for each class in a numpy array in one line while using the same covariance matrix self.weighted_cov
            prob_product = np.array([self.N(x, self.means[i], self.weighted_cov) * self.priors[i] for i in range(len(self.labels))])
        return prob_product/np.sum(prob_product) 
    
    def predict_proba(self, x_val:np.ndarray)->np.ndarray:
        # given x_val of dimensions (m,n) apply predict_proba_x to each row (point x) to return array of probabilities (m, k)
        return np.apply_along_axis(self.predict_proba_x, axis=1, arr=x_val)
    
    
    def predict(self, x_val:np.ndarray)->np.ndarray:
        # TODO 9: Get the final predictions by computing argmax over the result from self.predict_proba
        y_pred_prob = self.predict_proba(x_val)
        y_pred_inds = np.argmax(y_pred_prob, axis=1) #for each entry, axis=1
        # replace each prediction with label in self.labels
        y_pred = self.labels[y_pred_inds]
        return y_pred
    
    def score(self, x_val:np.ndarray, y_val:np.ndarray)->float:
        y_pred = self.predict(x_val)
        # TODO 10: compute accuracy in one line by comparing y_val and y_pred 
        #number pr correct predictions over the total number predictions
        acc = np.mean(y_pred==y_val) #[1, 0, 1, 1] 3/4
        return acc