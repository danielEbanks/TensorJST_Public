import numpy as np
import math
from scipy.stats import gamma

import tensorly as tl
from cumulant_gradient import cumulant_gradient
import tensor_lda_util as tl_util


class TLDA():
    def __init__(self, n_topic,n_senti ,alpha_0, n_iter_train, n_iter_test, batch_size, learning_rate = 0.001, gamma_shape = 1.0, smoothing = 1e-6): # we could try to find a more informative name for alpha_0
        # set all parameters here
        self.n_topic = n_topic
        self.n_senti = n_senti
        self.alpha_0 = alpha_0
        self.n_iter_train  = n_iter_train
        self.n_iter_test   = n_iter_test
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.gamma_shape = gamma_shape
        self.smoothing   = smoothing
        self.weights_ = tl.ones(self.n_topic*self.n_senti)
        self.factors_ = tl.tensor(np.random.randn(self.n_topic, self.n_topic)/10000)

    def partial_fit(self, X_batch, verbose = False):
        '''Update the factors directly from the batch using stochastic gradient descent
        Parameters
        ----------
        X_batch : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors
        verbose : bool, optional
            if True, print information about every 200th iteration
        '''
        # incremental version
        y_mean = tl.mean(X_batch, axis=0)

        for i in range(1, self.n_iter_train):
            for j in range(0, len(X_batch)-(self.batch_size-1), self.batch_size):
                y = X_batch[j:j+self.batch_size]

                lr = self.learning_rate*(10/(10+i))
                self.factors_ -= lr*cumulant_gradient(self.factors_, y, self.alpha_0,1)
            if verbose == True and (i % 200) == 0:
                print("Epoch: " + str(i) )

    def fit(self, X, verbose = False):
        '''Update the factors directly from X using stochastic gradient descent
        Parameters 
        ----------
        X : ndarray of shape (number_documents, num_topics) equal to the whitened
            word counts in each document in the documents used to update the factors
        '''

        self.partial_fit(X, verbose = verbose)

    def _predict_topic(self, doc, adjusted_factor,w_mat):
        '''Infer the document-topic distribution vector for a given document using variational inference
        Parameters
        ----------
        doc : tensor of length vocab_size equal to the number of occurrences
                      of each word in the vocabulary in a document
        adjusted_factor : tensor of shape (number_topics, vocabulary_size) equal to the learned
               document-topic distribution
        Returns
        -------
        gammad : tensor of shape (1, n_cols) equal to the document/topic distribution
                 for the doc vector
        '''
        
        n_cols = len(self.factors_)
        if w_mat == False:
            self.weights_ = self.weights_ = tl.ones(self.n_topic*self.n_senti)/(self.n_topic*self.n_senti)
        if w_mat == True:
            self.weights_ = self.weights_ = tl.ones(self.n_topic)/(self.n_topic)

        


        gammad = tl.tensor(gamma.rvs(self.gamma_shape, scale= 1.0/self.gamma_shape, size = n_cols)) # gamma dist. 
        exp_elogthetad = tl.tensor(np.exp(tl_util.dirichlet_expectation(gammad)))
        exp_elogbetad  = tl.tensor(np.array(adjusted_factor))

        phinorm = (tl.dot(exp_elogbetad, exp_elogthetad) + 1e-100)
        mean_gamma_change = 1.0

        iter = 0
        while (mean_gamma_change > 1e-2 and iter < self.n_iter_test):
            lastgamma = tl.copy(gammad)
            gammad = ((exp_elogthetad * (tl.dot(exp_elogbetad.T, doc / phinorm))) + self.weights_)
            exp_elogthetad = tl.tensor(np.exp(tl_util.dirichlet_expectation(gammad)))
            phinorm = (tl.dot(exp_elogbetad, exp_elogthetad) + 1e-100)

            mean_gamma_change = tl.sum(tl.abs(gammad - lastgamma)) / n_cols
            all_gamma_change = gammad-lastgamma
            iter += 1

        return gammad

    def predict(self, X_test,w_mat=False,doc_predict=True):
        '''Infer the document/topic distribution from the factors and weights and
        make the factor non-negative
        Parameters
        ----------
        X_test : ndarray of shape (number_documents, vocabulary_size) equal to the word
            counts in each test document
        Returns
        -------
        gammad_norm2 : tensor of shape (number_documents, number_topics) equal to
                       the normalized document/topic distribution for X_test
        factor : tensor of shape (vocabulary_size, number_topics) equal to the
                 adjusted factor
        '''

        adjusted_factor = tl.transpose(self.factors_)
        #adjusted_factor = tl_util.non_negative_adjustment(adjusted_factor)
        #adjusted_factor = tl_util.smooth_beta(adjusted_factor, smoothing=self.smoothing)
        
        
        # set negative part to 0
        adjusted_factor[adjusted_factor < 0.] = 0.
        # smooth beta
        adjusted_factor *= (1. - self.smoothing)
        adjusted_factor += (self.smoothing / adjusted_factor.shape[1])

        adjusted_factor /= adjusted_factor.sum(axis=1)[:, np.newaxis]
        
        
        if doc_predict == True:

            gammad_l = (np.array([tl.to_numpy(self._predict_topic(doc, adjusted_factor,w_mat)) for doc in X_test]))
            gammad_l = tl.tensor(np.nan_to_num(gammad_l))

            #normalize using exponential of dirichlet expectation
            gammad_norm = tl.tensor(np.exp(np.array([tl_util.dirichlet_expectation(g) for g in gammad_l])))
            gammad_norm2 = tl.tensor(np.array([row / np.sum(row) for row in gammad_norm]))

            return gammad_norm2, tl.transpose(adjusted_factor)
        else:
            return tl.transpose(adjusted_factor) 