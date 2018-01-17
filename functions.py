import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma 

def getDataDimensions(input_data):
    # gets the dimensions of the data tf
    # D Documents, N words
    return input_data.shape

def loadMockData(D,V,max_repeats=5):
    return np.random.randint(max_repeats, size=(D,V))

#tf = loadData()
tf = loadMockData(50, 100)

# Initialize parameters
    
# tf: input_data
K = 10 # Number of Topics
D = getDataDimensions(tf)[0] # Number of Documents
V = getDataDimensions(tf)[1] # Number of words in dictionary
N = np.max(np.sum(tf, axis = 1))
beta_init = np.zeros((D, N)) # betas
alpha_init = [50.0/K] * K # alphas
gamma_init = [alpha_init[0] + float(N)/K] * K # variational hypermeter of topic distribution
phi_init = np.full((K, N), 1./K) # variational hypermeter of topics
eta = 0.001 # fixed hyperparameter for the smoothed version
Lambda_init = np.zeros((D, N)) # variational hyperparameter for smoothed version
likelihood = 0 # Likelihood to be minimized
EstepConvergeThreshold = 10**(-5)
EstepMaxIterations = 10 


## Functions

def ComputeLikelihood(tf, K, D, N, V, alpha, eta, gamma, phi, Lambda):
    # Computes likelihood of the variational model (eq. 15).

    return(likelihood)


def ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda):
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
    likelihood = 10**(-10) 
    converged = False
    iterations = 0
    while(not(converged)):
        if(iterations > EstepMaxIterations):
            raise ValueError('E-step not converged after %d iterations' %iterations)
        
        iterations += 1
    
        for d in range(D):
            wd = np.repeat(range(V), tf[d,:])
            for n in range(len(wd)):
                for k in range(K):
                    phi[d,n,k] = np.exp(digamma(gamma[d,k]) - digamma(np.sum(gamma[d,:])) + digamma(Lambda[k,wd[n]]) - digamma(np.sum(Lambda[k,:])))
                phi[d,n,:] /= sum(phi[d,n,:])
    
        for d in range(D):
            gamma[d,:] = alpha[d] + np.sum(phi[d,:,:], axis = 0)
        
        # newLikelihood = ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda)
        newLikelihood = 0.1
        dlikelihood = (newLikelihood - likelihood)/likelihood
        likelihood = newLikelihood
        if(dlikelihood < EstepConvergeThreshold):
            print('E-step converged after %d iterations' %iterations)
            break
        
    Lambda[k,:] = eta[k]
    for d in range(D):
        wd = np.repeat(range(V), tf[d,:])
        for k in range(K):
            for n in range(len(wd)):        
                Lambda[k,wd[n]] += phi[d,n,k]
                
    return(phi, gamma, Lambda, likelihood)


def ExpectationStepUnitTest():
    alpha = np.random.rand(D,1)
    eta = np.random.rand(K,1)
    gamma = np.random.rand(D,K)
    phi = np.random.rand(D,N,K)
    Lambda = np.random.rand(K,V)
    
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)

ExpectationStepUnitTest()

def updateAlpha(D, K, alpha, gamma):
    # Newton-Rhaphson algorithm for updating alpha
    alpha = np.random.rand(D,1)
    return(alpha)
    
def updateEta(V, K, Lambda, eta): 
     # Newton-Rhaphson algorithm for updating alpha
    eta = np.random.rand(V, 1)
    return(eta)


def MaximizationStep(D, V, K,  alpha, gamma, phi, Lambda, eta, likelihood):
    # Returns updated alpha, eta
    alpha = updateAlpha(D, K, alpha, gamma)
    
    eta = updateEta(V, K, Lambda, eta)
    
    return(alpha, eta)


def VariationalExpectationMaximization( ):
    # Calculates variational parameters gamma, phi, and lambda iteratively until convergence
    
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)
    
    alpha, eta = MaximizationStep(D, V, K, alpha, gamma, phi, Lambda, eta, likelihood)
    
    return(gamma, Lambda, phi)
    
    
    
    
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


## Document Classifiaction
    
def DocumentClassification(tf, gamma):
    # Performs SVM for binary classification based on 1) original count data and
    #  2) per document-topic composition
    return(accuracy) # accuracy: #Iterations I x 2 (Accuracy of Word features, accuracy of LDA features)



def plotSVMModelAccuracy(accuracy):
    # Training data - accuracy plot (fig. 10)
    plt.errorbar(np.arange(0.1,0.35,0.05),np.arange(70, 95, 5), 
                 yerr = 2, label="Word features", fmt="s--", linewidth=1)
    plt.errorbar(np.arange(0.1,0.35,0.05),np.arange(95, 70, -5), yerr= 2, label="LDA features", fmt="s-", linewidth=1)
    plt.rc('axes', labelsize = 15)
    plt.xlabel('Proportion of data used for training')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', shadow=True, fontsize='x-large', prop={'size': 10})
    plt.show()
    


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
