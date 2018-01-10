import numpy as np
import matplotlib.pyplot as plt


def getDataDimensions(input_data):
    # gets the dimensions of the data tf
    # D Documents, N words
    return (D,N)


# Initialize parameters
    
# tf: input_data
K = 10 # Number of Topics
D = getDataDimensions(tf)[1] # Number of Documents
N = getDataDimensions(tf)[2] # Number of words
beta_init = np.zeros((D, N)) # betas
alpha_init = [50/K] * K # alphas
gamma_init = [alpha[1] + N/K] * K # variational hypermeter of topic distribution
phi_init = np.full((K, N), 1./K) # variational hypermeter of topics
eta = 0.001 # fixed hyperparameter for the smoothed version
Lambda_init = np.zeros((D, N)) # variational hyperparameter for smoothed version
likelihood =  0 # Likelihood to be minimized
converge_threshold = 0.0001 # Converge criterion


## Functions

def ComputeLikelihood(tf, alpha, beta, gamma, phi, Lambda):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the betas. (smoothed version) 
    return(likelihood)


def ExpectationStep(K, D, N, alpha, beta, gamma, phi, Lambda):
    # Returns updated phi, gamma and likelihood L(γ,φ,λ; α,β)
    return(phi, gamma, likelihood)


def updateAlpha(D, K, alpha, gamma):
    # Newton-Rhapson algorithm for updating alpha
    return(alpha)


def MaximizationStep(tf, K, D, N, phi, gamma, likelihood):
    # Returns updated alpha, beta and lambda
    return(alpha, beta, Lambda)


def ExpectationMaximization( ):
    # Calculates variational parameters gamma, phi, and lambda iteratively until convergence
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
