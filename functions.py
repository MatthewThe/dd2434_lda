# -*- coding: utf-8 -*-

import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.special import digamma, polygamma, gammaln 
from scipy import special, misc
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer
import os
from bs4 import BeautifulSoup

import zipfile
import os.path

from sklearn import datasets, svm

def main(argv):
    np.random.seed(1)
    K = 10
    dataSet = "cancer" # one of "simulated", "reuters", "cancer", "simple"
        
    print("*** LOADING DATA ***")
    if dataSet == "simulated":
        D = 50
        N = 15
        V = 30
        #K_generate = 5
        K_generate = K
        alpha_original, eta_original, beta_original, theta_original, Z_original, _, tf = createSampleSparse(D,N,K_generate,V,maxAlpha=1, maxEta=1)
        #tf = tf.toarray()
        #print tf
        print("*** Original alpha: %.5f, Original eta: %.5f ***\n" % (alpha_original, eta_original))
        print("*** Original theta ***")
        print(theta_original)
        print("*** Original beta ***")
        print(beta_original)
        
        alpha_init = 5.0 / K
        eta_init = 1.0
    elif dataSet == "reuters":
        tf, labels_1, labels_2, topic_texts, dictionary = loadData('reutersdata', 'earn', 'grain', maxDocs = 8000)
        alpha_init = 50.0 / K
        eta_init = 50.0 / len(dictionary)
    elif dataSet == "simple":
        tf = loadSimpleData()
        tf = csr_matrix(tf)
        alpha_init = 5.0 / K
        eta_init = 1.0
    elif dataSet == "cancer":
        filename = "data_CNA_notna.txt"
        tf = loadCancerDataPosNeg(filename) # or tf=loadCancerDataAbsolute(filename) or tf=loadCancerDataBinary(filename)
        alpha_init = 1.0
        eta_init = 5.0
    else:
        sys.exit("Unknown dataset: %s" % dataSet)
    
    # Initialize parameters

    # tf: input_data
    #K = 10 # Number of Topics
    D = getDataDimensions(tf)[0] # Number of Documents
    V = getDataDimensions(tf)[1] # Number of words in dictionary
    N = np.max(tf.sum(axis=1))
    
    print("Num documents: %d, Dictionary size: %d, Max words in document: %d, Num topics: %d" % (D, V, N, K))

    #ExpectationStepUnitTest()
    #MaximizationStepUnitTest()
    #VariationalExpectationMaximizationUnitTest(tf, D, K, V, N)
    
    gamma = np.ones((D,K)) * (alpha + float(N)/K)
    Lambda = np.random.rand(K,V) * 0.5 + 0.5
    phi = np.ones((D,N,K)) * (1./ K)
    
    gamma, Lambda, phi = VariationalExpectationMaximization(tf, alpha_init, eta_init, gamma, phi, Lambda) 

    if dataSet == "simulated":
        print("*** Original alpha: %.5f, Original eta: %.5f ***\n" % (alpha_original, eta_original))
        print("\n*** Original beta ***")
        print(beta_original)
        print("*** Original theta vs VEM gamma ***")
        compareGamma = gamma - alpha
        for k in range(K_generate):
            compareGamma = np.concatenate((compareGamma, np.sum(Z_original == k, axis = 1)[np.newaxis,:].T), axis = 1)
        print(compareGamma)
        plt.imshow(compareGamma, aspect = 'auto', interpolation = 'none', cmap = 'Reds')
        plt.show()
    elif dataSet == "reuters":
        printResultsReuters(tf, gamma, Lambda, phi, topic_texts)
    elif dataSet == "simple":
        print(tf)
        print(gamma)
    elif dataSet == "cancer":
        printResults(tf, gamma, Lambda, phi)
    else:
        sys.exit("Unknown dataset: %s" % dataSet)
        
def printResultsReuters(tf, gamma, Lambda, phi, topic_texts):
    D, K = gamma.shape
    writer = csv.writer(open('output_gamma.tsv', 'w'), delimiter = '\t')
    for d in range(D):
        writer.writerow(list(gamma[d,:]) + [topic_texts[d]])
    
    writer = csv.writer(open('output_Lambda.tsv', 'w'), delimiter = '\t')
    K, V = Lambda.shape
    for k in range(K):
        writer.writerow(list(Lambda[k,:]))
    
    writer = csv.writer(open('output_phi.tsv', 'w'), delimiter = '\t')
    D, N, K = phi.shape
    for d in range(D):
        writer.writerow([d])
        for n in range(int(tf[d,:].sum())):
            writer.writerow(list(phi[d,n,:]))


def printResults(tf, gamma, Lambda, phi):
    D = gamma.shape[0]
    writer = csv.writer(open('output_gamma.tsv', 'w'), delimiter = '\t')
    for d in range(D):
        writer.writerow(list(gamma[d,:]))
    
    writer = csv.writer(open('output_Lambda.tsv', 'w'), delimiter = '\t')
    K, V = Lambda.shape
    for k in range(K):
        writer.writerow(list(Lambda[k,:]))
    
    writer = csv.writer(open('output_phi.tsv', 'w'), delimiter = '\t')
    D, N, K = phi.shape
    for d in range(D):
        writer.writerow([d])
        for n in range(int(tf[d,:].sum())):
            writer.writerow(list(phi[d,n,:]))
    
##### Start of Synthetic Data Generation #####
def createAlpha(maxAlpha=1):
    alpha = np.random.rand() * maxAlpha
    return alpha

def createEta(maxEta=1):
    eta = np.random.rand() * maxEta
    return eta

def createBeta(eta, V, K):
    eta_vector = np.ones((V)) * eta
    beta = np.random.dirichlet(eta_vector, K)
    return beta

def createTheta(alpha, K, D):
    alpha_vector = np.ones((K)) * alpha
    theta = np.random.dirichlet(alpha_vector, D)
    return theta

def createZ(theta, D, N, K):
    Z = np.zeros((D,N,K))
    for d in range(D):
        Z[d] = np.random.multinomial(1, theta[d], size=N)
    Z = Z.astype(int)
    return Z

def createW(beta, Z, D, N, V):
    w = np.zeros((D,N,V))
    for d in range(D):
        for n in range(N):
            z_dn = Z[d,n]
            k = np.where(z_dn==1)[0][0]
            w[d,n] = np.random.multinomial(1, beta[k], size=1)
            
    w_counts = np.sum(w, axis=1) # D x V, all the entries are the number of words occurred in the document. 
    w = w.astype(int)
    w_counts = w_counts.astype(int)    
    return w, w_counts

def createSample(D,N,K,V,maxAlpha=1, maxEta=1):
    alpha = createAlpha(maxAlpha)
    eta = createEta(maxEta)
    beta = createBeta(eta, V, K)
    theta = createTheta(alpha, K, D)
    Z = createZ(theta, D, N, K)
    w, w_counts = createW(beta, Z, D, N, V)
    
    return alpha, eta, beta, theta, Z, w, w_counts
    
    
def createZNew(theta, D, N):
    Z = np.zeros((D,N))
    for d in range(D):
        temp = np.random.multinomial(1, theta[d], size=N)
        topicLabels = np.where(temp==1)[1]
        Z[d] = topicLabels
    Z = Z.astype(int)
    return Z

def createWNew(beta, Z, D, V, K):
    w = csr_matrix((D, V), dtype=np.int64)
    for d in range(D):  
        for k in range(K):
            kLabeledWords = np.where(Z[d]==k)[0] 
            count = len(kLabeledWords)
            
            temp = np.random.multinomial(1, beta[k], size=count)
            marg = np.sum(temp, axis=0)
            
            w[d,:] = w[d,:] +  marg
    w.eliminate_zeros()
    return w

def createSampleSparse(D,N,K,V,maxAlpha=1, maxEta=1):
    alpha = createAlpha(maxAlpha)
    eta = createEta(maxEta)
    beta = createBeta(eta, V, K)
    theta = createTheta(alpha, K, D)
    Z = createZNew(theta, D, N)
    w = createWNew(beta, Z, D, V, K)
    
    return alpha, eta, beta, theta, Z, w, w
    
##### End of Synthetic Data Generation #####

##### Start of Cancer Data Generation #####

def loadCancerDataBinary(filename):
# This function returns a binary (DxV) sparse matrix 
# where each row is a patient, each column is a gene
# and each entry represents an event (entry is 1 if copy number is (-2,-1,1 or 2)).
#
# filename = "data_CNA_notna.txt"
# tf = loadCancerDataBinary(filename)
    filepath = "CancerData/" + filename
    status = os.path.isfile(filepath) 
    if(not status):
        zipp = zipfile.ZipFile("./data_CNA_notna.txt.zip")
        zipp.extractall("CancerData")
        
    with open(filepath) as infile:
        # Read header line
        first_line = infile.readline()

        numColumns = len(first_line.split("\t"))
        numPatients = numColumns - 2 # Except first two columns

        # Read remaning lines
        linenum = 0
        rows = []
        cols = []
        for line in infile:
            contents = line.split()[2:]

            contents = np.array(list(map(int, contents)))
            nonzeros = np.where(contents != 0)[0]

            for n in range(nonzeros.shape[0]):
                rows.append(nonzeros[n]) # patients who have the words
                cols.append(linenum)     # word id

            linenum = linenum + 1

    numGenes = linenum 
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones((rows.shape[0]))

    geneMatrix = csr_matrix((data, (rows, cols)), shape=(numPatients, numGenes))
    return geneMatrix

def loadCancerDataAbsolute(filename):
# This function returns a (DxV) sparse matrix 
# where each row is a patient, each column is a gene
# and each entry represents an event (entry is 1 if copy number is (-1 or 1) or 2 if copy number is (-2 or 2)).
#
# filename = "data_CNA_notna.txt"
# tf = loadCancerDataAbsolute(filename)

    filepath = "CancerData/" + filename
    status = os.path.isfile(filepath) 
    if(not status):
        zipp = zipfile.ZipFile("./data_CNA_notna.txt.zip")
        zipp.extractall("CancerData")

    with open(filepath) as infile:
        # Read header line
        first_line = infile.readline()

        numColumns = len(first_line.split("\t"))
        numPatients = numColumns - 2 # Except first two columns

        # Read remaning lines
        linenum = 0
        rows = []
        cols = []
        data = []
        for line in infile:
            contents = line.split()[2:]

            contents = np.array(list(map(int, contents)))
            nonzeros = np.where(contents != 0)[0]

            for n in range(nonzeros.shape[0]):
                rows.append(nonzeros[n]) # patients who have the words
                cols.append(linenum)     # word id
                data.append(np.abs(contents[nonzeros[n]]))
                
            linenum = linenum + 1

    numGenes = linenum 
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    geneMatrix = csr_matrix((data, (rows, cols)), shape=(numPatients, numGenes))
    return geneMatrix

def loadCancerDataPosNeg(filename):
# This function returns a (Dx2V) sparse matrix 
# where each row is a patient, 
# each column is a gene (we set 2 columns per gene; gene_neg and gene_pos to represent dropout and amplification events)
# and each entry represents an event (entry is 1 or 2, its column depends on the sign (- or +)).
#
# filename = "data_CNA_notna.txt"
# tf = loadCancerDataPosNeg(filename)
    filepath = "CancerData/" + filename
    status = os.path.isfile(filepath) 
    if(not status):
        zipp = zipfile.ZipFile("./data_CNA_notna.txt.zip")
        zipp.extractall("CancerData")

    with open(filepath) as infile:
        # Read header line
        first_line = infile.readline()

        numColumns = len(first_line.split("\t"))
        numPatients = numColumns - 2 # Except first two columns

        # Read remaning lines
        linenum = 0
        rows = []
        cols = []
        data = []
        for line in infile:
            contents = line.split()[2:]

            contents = np.array(list(map(int, contents)))
            nonzeros = np.where(contents != 0)[0]

            for n in range(nonzeros.shape[0]):
                val = contents[nonzeros[n]]
                
                rows.append(nonzeros[n]) 
                if val<0:
                    cols.append(2*linenum) # negative words (lost the gene)
                else:
                    cols.append(2*linenum+1) # positive words (have multiple genes)
                data.append(np.abs(val))
                
            linenum = linenum + 1

    numGenes = linenum 
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    geneMatrix = csr_matrix((data, (rows, cols)), shape=(numPatients, 2*numGenes))
    return geneMatrix
##### End of Cancer Data Generation #####

def getDataDimensions(input_data):
    # gets the dimensions of the data tf
    # D Documents, N words
    return input_data.shape

def loadMockData(D,V,max_repeats=5):
    return np.random.randint(max_repeats, size=(D,V))


def loadSimpleData():
    return np.array([[9,1,9],[1,9,1],[9,1,9],[10,10,10],[5,5,11]])

## Functions

def ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the etas. (smoothed version) 
    
    K, V = Lambda.shape
    D, N = tf.shape
    
    digamma_lambda = digamma(Lambda.T) - digamma(np.sum(Lambda, axis = 1))
    Likelihood = np.zeros((D,1))
    for d in range(D):
        wd = wordCountToFlatArray(tf[d,:])
        digamma_gamma = digamma(gamma[d,:]) - digamma(np.sum(gamma[d,:]))
        Likelihood[d] = ComputeDocumentLikelihood(wd, alpha, eta, gamma, digamma_gamma, phi, digamma_lambda, d)
    
    #Hazal's notes, eq. 8
    E_beta_eta = K * (gammaln(eta*V) - V*gammaln(eta)) + (eta - 1) * np.sum(digamma_lambda)
    
    #Hazal's notes, eq. 12
    E_beta_lambda = np.sum(gammaln(np.sum(Lambda, axis = 1)) - np.sum(gammaln(Lambda), axis = 1)[np.newaxis,:]) \
                    + np.sum((Lambda - 1) * digamma_lambda.T)
    
    #print "plus", E_beta_eta
    #print "min", E_beta_lambda
    Likelihood = np.sum(Likelihood) + E_beta_eta - E_beta_lambda
    return(Likelihood)

def ComputeDocumentLikelihood(wd, alpha, eta, gamma, digamma_gamma, phi, digamma_lambda, d):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the betas. (smoothed version) 
    
    V, K = digamma_lambda.shape
    N = len(wd)
    
    #Hazal's notes, eq. 6
    E_theta_alpha = gammaln(alpha*K) - K * gammaln(alpha) \
                        + (alpha-1) * np.sum(digamma_gamma)
    
    #Hazal's notes, eq. 7
    E_z_theta = np.dot(np.sum(phi[d,:N,:], axis = 0), digamma_gamma)
    
    #Hazal's notes, eq. 9
    E_w_z_beta = np.sum(digamma_lambda[wd,:] * phi[d,:N,:])
    
    #Hazal's notes, eq. 10
    E_theta_gamma = gammaln(np.sum(gamma[d,:])) - np.sum(gammaln(gamma[d,:])) \
                    + np.dot(gamma[d,:] - 1, digamma_gamma)

    #Hazal's notes, eq. 11
    E_z_phi = np.sum(phi[d,:N,:] * np.log(phi[d,:N,:]))
    
    #print "plus", E_theta_alpha, E_z_theta, E_w_z_beta
    #print "min", E_theta_gamma, E_z_phi
    
    Likelihood = E_theta_alpha + E_z_theta + E_w_z_beta - E_theta_gamma - E_z_phi
    return(Likelihood)

def wordCountToFlatArray(tf):
    return np.repeat(tf.indices, tf.data)
    #return np.repeat(range(40), tf)

def ExpectationStep(tf, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 1e-4, EstepMaxIterations = 100):
    # Returns updated phi, gamma and likelihood L(γ,φ,λ; α,β)
    #
    # k: topic (N.B.: r in Hazal's notes)
    # d: document
    # n: word in document
    #
    # γ: [document, topic], topic probabilities per document (Dirichlet)
    # λ: [topic, word in dictionary], word emission probability per topic (Dirichlet)
    # φ: [document, word in document, topic], word and topic probability for each word in the document (Multinomial)
    # tf: [document, word in dictionary]
    # α: [document], prior topic probability
    # eta: [topic], prior word probability 
    
    K, V = Lambda.shape
    D, N = tf.shape
    
    iterations = 0
    likelihood = -1e9
    converged = False
    while(not(converged)):
        if(iterations > EstepMaxIterations):
            raise ValueError('E-step not converged after %d iterations' %iterations)
        
        iterations += 1
        phi, gamma = ExpectationPhiGamma(tf, alpha, eta, gamma, phi, Lambda)
        #print("before", ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda), Lambda)
        
        # update Lambda
        Lambda[:,:] = eta
        for d in range(D):
            wd = wordCountToFlatArray(tf[d,:])
            for n in range(len(wd)):        
                Lambda[:,wd[n]] += phi[d,n,:]
        
        newLikelihood = ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda)
        dlikelihood = abs((newLikelihood - likelihood)/likelihood)
        #print("after", newLikelihood, Lambda)
        if newLikelihood < likelihood and dlikelihood > EstepConvergeThreshold:
            print('WARNING: E-step lower bound decreased')
            print("old", likelihood, "new", newLikelihood)
            sys.exit()
        likelihood = newLikelihood
        print("Likelihood after E-step iteration: %f" % likelihood)
        if(dlikelihood < EstepConvergeThreshold):
            print('E-step converged after %d iterations' %iterations)
            converged = True
    return(phi, gamma, Lambda, likelihood)

def ExpectationPhiGamma(tf, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 1e-4, EstepMaxIterations = 100):
    K, V = Lambda.shape
    D, _ = tf.shape
    
    digamma_lambda = digamma(Lambda.T) - digamma(np.sum(Lambda, axis = 1))
    for d in range(D):
        if d % 500 == 0:
            print("E-step phi gamma document %d" % d)
        
        likelihood = -1e9
        converged = False
        iterations = 0
        while(not converged):
            if(iterations > EstepMaxIterations):
                raise ValueError('E-step phi gamma not converged after %d iterations' %iterations)
            iterations += 1
            wd = wordCountToFlatArray(tf[d,:])
            
            digamma_gamma = digamma(gamma[d,:]) - digamma(np.sum(gamma[d,:]))
            
            N = len(wd)
            
            phi[d,:N,:] = digamma_gamma + digamma_lambda[wd,:]
            phi[d,:N,:] = np.exp(phi[d,:N,:] - misc.logsumexp(phi[d,:N,:], axis = 1)[:,np.newaxis])
            
            gamma[d,:] = alpha + np.sum(phi[d,:N,:], axis = 0)
            
            newLikelihood = ComputeDocumentLikelihood(wd, alpha, eta, gamma, digamma_gamma, phi, digamma_lambda, d)
            dlikelihood = abs((newLikelihood - likelihood)/likelihood)
            if newLikelihood < likelihood and dlikelihood > EstepConvergeThreshold:
                print('WARNING: E-step phi gamma lower bound decreased')
                print("old", likelihood, "new", newLikelihood)
                sys.exit()
            likelihood = newLikelihood
            #print(likelihood)
            if(dlikelihood < EstepConvergeThreshold):
                #print('E-step phi gamma converged after %d iterations' %iterations)
                converged = True
    
    #print gamma
    return(phi, gamma)   

def ExpectationStepUnitTest():
    alpha = np.random.rand()
    eta = np.random.rand()
    gamma = np.random.rand(D,K)
    phi = np.random.rand(D,N,K)
    Lambda = np.random.rand(K,V)
    
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, alpha, eta, gamma, phi, Lambda)
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, alpha, eta, gamma, phi, Lambda)

### Start of Maximization Part ###

def computeSufficientStats(D, K, gamma):
    # Given the parameter (gamma or Lambda), this function calculates the sufficient statistics.
    stats = 0
    sum_gamma = np.sum(gamma, axis=1)   # (D x 1)
    for d in range(D):
        stats = stats - K * polygamma(0, sum_gamma[d]) # digamma
        di_gamma_d = polygamma(0, gamma[d])
        stats = stats + np.sum(di_gamma_d)
        
    return stats
    
def computeL(D, K, alpha, stats):
    # Given the parameter (alpha or eta) and the sufficient statistics, this function calculates the lower bound.
    lik = D * gammaln(K*alpha) 
    lik = lik - D * K * gammaln(alpha)
    lik = lik + (alpha - 1) * stats
    return lik

def computeGradient(D, K, alpha, stats):
    # Given the parameter pairs (alpha&gamma or eta&Lambda), this function calculates the gradient.
    gradient = D * K * polygamma(0, K*alpha) # digamma
    gradient = gradient - D * K * polygamma(0, alpha)
    gradient = gradient + stats
    return gradient

def computeHessian(D, K, alpha):
    # Given the parameter (alpha or eta), this function calculates the hessian.
    hessian = D * K * K * polygamma(1, K*alpha) # trigamma
    hessian = hessian - D * K * polygamma(1, alpha)
    return hessian

def moveAlpha(alpha, gradient, hessian):
    # This function updates the parameter.
    # The logarithmic scale is taken from Colorado Reed paper.
    invHg = gradient / (hessian * alpha + gradient)
    
    log_alpha_new = np.log(alpha) - invHg
    alpha_new = np.exp(log_alpha_new)
    
    # OR!!
    #alpha_new = alpha - invHg
    
    return alpha_new

def computeNegL(alpha,D,K,stats):
    # This function calculates the Negative Log Likelihood of Alpha related terms
    return - computeL(D, K, alpha, stats)
    
def updateAlpha_Scipy(alpha, gamma):
    # This function minimizes the Negative Log Likelihood
    D, K = gamma.shape
    stats = computeSufficientStats(D, K, gamma)

    if np.isnan(alpha):
        print("ALPHA WAS NAN")
        alpha = 10
    else:   
        negBound = computeNegL(alpha,D,K,stats)
        res = minimize(computeNegL, alpha, args=(D,K,stats))

        newnegBound = computeNegL(res.x[0],D,K,stats)
        #print("\tAlpha:%.5f. negL: %.5f. New Alpha: %.5f. New-negL:%.5f" % (alpha,negBound,res.x[0], newnegBound))
        alpha = res.x[0]
        
    return alpha

def updateAlpha(alpha, gamma, maxIter=100, tol=0.0001):
    # This function updates alpha OR eta parameter. 
    # Since the form of equations are the same, we can use this function to update eta as well.
    #
    # param1: alpha (scalar) OR      eta (scalar)
    # param2: gamma (DxK)    OR   Lambda (KxV)
    # maxIter & tol: Convergence criteria.
    isConverged = False
    
    D, K = gamma.shape
    
    epoch = 0
    bounds = []
    
    stats = computeSufficientStats(D, K, gamma)
    
    init_alpha = alpha
    init_L = computeL(D, K, alpha, stats)
    
    while(not isConverged):
        
        if np.isnan(alpha):
            print("ALPHA WAS NAN. Re-initializing")
            alpha = 10
        else:
            bound = computeL(D, K, alpha, stats)
            
            if np.isinf(bound) or np.isnan(bound):
                print("Lower bound Problem (INF or NAN)")
                print("alpha")
                print(alpha)
                print("bound")
                print(bound)
                print("stats")
                print(stats)
                print("gamma")
                print(gamma)
                sys.exit()
                
            gradient = computeGradient(D, K, alpha, stats)
            hessian = computeHessian(D, K, alpha)
    
            #print("Epoch: %d" %epoch)
            #print("\tValue:%.5f. L: %.5f. Gradient: %.5f. Hessian: %.5f" % (alpha,bound,gradient,hessian))
            alpha = moveAlpha(alpha, gradient, hessian)
            
            #print("New likelihood:")
            #print(newLikelihood)
            #print("\tNew Value: %.5f" % (alpha))
            
            bounds.append(bound)
            epoch = epoch + 1
            
            if epoch==maxIter:
                isConverged = True
                print("\t\tConverged: max iteration")
            else:
                if epoch>1 and np.absolute(gradient)<tol:
                    isConverged = True
                    print("\t\tConverged: gradient close to zero at iteration: %d" % epoch)
                    
            #if(isConverged):
            #    print("\tAlpha:%.5f. L:%.5f. New Alpha:%.5f. newL: %.5f. " % (init_alpha, init_L, alpha,computeL(D, K, alpha, stats)))
    
    return alpha

def MaximizationStep(alpha, gamma, eta, Lambda):
    # Returns updated alpha and eta

    # print("\tM-Step")

    #gamma = np.random.rand(D,K) * 10 +10    
    
    alpha = updateAlpha(alpha, gamma)
    #alpha = updateAlpha_Scipy(alpha, gamma)
    
    eta  = updateAlpha(eta, Lambda)
    #eta = updateAlpha_Scipy(eta, Lambda)
    
    return(alpha, eta)

def MaximizationStepUnitTest():
    alpha = np.random.rand()
    gamma = np.random.rand(D,K)
    eta = np.random.rand() 
    Lambda = np.random.rand(K,V)
    
    alpha, eta = MaximizationStep(alpha, gamma, eta, Lambda)

### End of Maximization Part ###

def VariationalExpectationMaximization(tf, alpha, eta, gamma, phi, Lambda, convergeThreshold = 1e-6, maxIterations = 500):
    print("*** VARIATIONAL EM ***")
    
    # Calculates variational parameters gamma, phi, and lambda iteratively until convergence
    likelihood = 10**(-10) 
    converged = False
    iterations = 0
    while(not converged):
        iterations += 1
        phi, gamma, Lambda, newLikelihood = ExpectationStep(tf, alpha, eta, gamma, phi, Lambda)
        #print("gamma")
        #print(gamma)
        
        print("\tM-Step")
        print("\t\tAlpha:%.5f. Eta: %.5f. Complete Likelihood: %.5f." % (alpha, eta, newLikelihood))
        alpha, eta = MaximizationStep(alpha, gamma, eta, Lambda)
        print("\t\tNew Alpha:%.5f. New Eta: %.5f. New Complete Likelihood: %.5f. " % (alpha, eta, ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda)))
        
        dlikelihood = abs((newLikelihood - likelihood)/likelihood)
        print("DeltaLikelihood: %g" % dlikelihood)
        likelihood = newLikelihood
        #print("gamma")
        #print(gamma)
        if(dlikelihood < convergeThreshold):
            print('EM-step converged after %d iterations' %iterations)
            converged = True
    return(gamma, Lambda, phi)
    
def VariationalExpectationMaximizationUnitTest(tf, D, K, V, N):
    alpha = 5.0 / K
    gamma = np.random.rand(D,K)
    eta = 5.0 # np.random.rand() 
    Lambda = np.random.rand(K,V)
    phi = np.random.rand(D,N,K)
    for d in range(D):
        for n in range(N):
            phi[d,n,:] /= np.sum(phi[d,n,:])
    
    VariationalExpectationMaximization(tf, alpha, eta, gamma, phi, Lambda)   

# Document modeling
    
def pWUnseenDocument(pW, K):
    # computes probability of unseen document
    # pW: a matrix with N * K, N: words, K: topics 
    return(prob_documents)
    

def computePerplexityDocModel(prob_documents, Nw):
    # computes perplexity for document modeling
    # prob_documents: unseen document probabilities (#Unseen of documents x 1 vector)
    # Nw: #words per document
    return(perp)      


def plotComplexityDocumentModeling(K, perp):
    # Topic - perplexity plot (fig. 9)
    plt.plot(np.arange(0, 60, 10), np.arange(650, 350, -50), marker = 'o', linestyle = '--', label='LDA', color = 'b')
    plt.rc('axes', labelsize = 15)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()


### Start of document classifiaction ###

# get 8000 documents out of ~21000 documents
def loadData(folderName, keyword_1, keyword_2, maxDocs = 1000, returnData = False):
    data = []
    labels_1 = []
    topic_of_interest_1 = keyword_1
    labels_2 = []
    topic_of_interest_2 = keyword_2
    topic_texts = []
    for file in os.listdir(folderName):
        if file.endswith('.sgm'):
            f = open(folderName + '/' + file, 'rb')
            # print('reutersdata/' + file)
            filestring = f.read()
            soup = BeautifulSoup(filestring, 'lxml')
            contents = soup.findAll('text')
            for content in contents:
                data.append(content.text)
            topics = soup.findAll('topics')
            for topic in topics:
                labels_1.append(topic_of_interest_1 in topic.text)
                labels_2.append(topic_of_interest_2 in topic.text)
                topic_texts.append(topic.text)
        if len(data) >= maxDocs:
            break
    vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                 token_pattern= '(?u)\\b[a-zA-Z_][a-zA-Z_]+\\b',
                                 stop_words='english')
    tf = vectorizer.fit_transform(data)
    dictionary = vectorizer.get_feature_names()
    
    if returnData:
      return (tf, labels_1, labels_2, topic_texts, dictionary, data)
    else:
      return (tf, labels_1, labels_2, topic_texts, dictionary)
    
### End of classificatiob ####    

    


## Collaborative filtering
    
def CollaborativeFiltering( ):
    # Estimates probability of unseen word by Variational Inference
    return(unseen_word_prob) # #Topics K x #Documents D matrix 


def PredictiveComplexity(unseenWordProbalities):
    # Calculates predictive complexity in collaborative filtering (section 7.3)
    # unseenWordProbabilities: #documents D * 1 vector
    return(complexity) #Topics *  1 vector

def plotPredictiveComplexity():
    # Topics - Predictive perplexity plot (fig. 11)
    plt.plot(np.arange(0, 60, 10), np.arange(650, 350, -50), marker = 'o', linestyle = '--', label='LDA', color = 'b')
    plt.rc('axes', labelsize = 15)
    plt.xlabel('Number of Topics')
    plt.ylabel('Predictive Perplexity')
    plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
