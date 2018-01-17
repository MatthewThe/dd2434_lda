import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma, gammaln 
from scipy import special
from sklearn.feature_extraction.text import CountVectorizer
import os
from bs4 import BeautifulSoup

def getDataDimensions(input_data):
    # gets the dimensions of the data tf
    # D Documents, N words
    return input_data.shape

def loadMockData(D,V,max_repeats=5):
    return np.random.randint(max_repeats, size=(D,V))

def loadData(folderName, keyword_1, keyword_2):
    data = []
    labels_1 = []
    topic_of_interest_1 = keyword_1
    labels_2 = []
    topic_of_interest_2 = keyword_2
    for file in os.listdir(folderName):
        if file.endswith('.sgm'):
            f = open(folderName + '/' + file,'rb')
            #print('reutersdata/' + file)
            filestring = f.read()
            soup = BeautifulSoup(filestring)
            contents = soup.findAll('text')
            for content in contents:
                data.append(content.text)
            topics = soup.findAll('topics')
            for topic in topics:
                labels_1.append(topic_of_interest_1 in topic.text)
                labels_2.append(topic_of_interest_2 in topic.text)
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(data)
    
    return(tf, labels_1, labels_2)

#tf, labels_1, labels_2 = loadData('reutersdata', 'earn', 'grain')
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

def ComputeLikelihood(tf, alpha, beta, gamma, phi, Lambda):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the betas. (smoothed version) 

    Likelihood = np.zeros((D,1))
    for d in range(D):
        #Hazal's notes, eq. 6
        E_theta_alpha = 0
        E_theta_alpha = gammaln(np.sum(alpha[d,:])) \
                        - np.sum(gammaln(alpha[d,:])) \
                        + np.sum([((alpha[d] - 1) \
                            *(digamma(gamma[d, i]) - digamma(np.sum(gamma[d,:])))) \
                           for i in range(K)])

        #Hazal's notes, eq. 7
        E_z_theta = 0
        wd = np.repeat(range(V), tf[d,:])
        for n in range(len(wd)):
            for r in range(K):
               E_z_theta += phi[d,n,r] *(digamma(gamma[d,r]) - digamma(np.sum(gamma[d,:])))

        #Hazal's notes, eq. 8
        E_beta_eta = 0
        for r in range(K):
            for i in range(V):
                E_beta_eta += gammaln(eta*V) - V*gammaln(eta) \
                              + (eta - 1)*(digamma(Lambda[r,i]) - digamma(np.sum(Lambda[r,:])))

        #Hazal's notes, eq. 9
        E_w_z_beta = 0
        wd = np.repeat(range(V), tf[d,:])
        for n in range(len(wd)):
            for r in range(K):
                for j in range(V):
                    E_w_z_beta += phi[d,n,r]*(digamma(Lambda[r,wd[n]]) - digamma(np.sum(Lambda[r,:])))

        #Hazal's notes, eq. 10
        E_theta_gamma = 0
        E_theta_gamma = gammaln(np.sum(gamma[d,:])) \
                        - np.sum(gammaln(gamma[d,:])) \
                        + np.sum([((gamma[d,i] - 1) \
                            *(digamma(gamma[d, i]) - digamma(np.sum(gamma[d,:])))) \
                           for i in range(K)])

        #Hazal's notes, eq. 11
        E_z_phi = 0
        wd = np.repeat(range(V), tf[d,:])
        for n in range(len(wd)):
            for r in range(K):
                E_z_phi += phi[d,n,r]*np.log(phi[d,n,r])

        #Hazal's notes, eq. 12
        E_beta_lambda = 0
        for r in range(K):
            E_beta_lambda += gammaln(np.sum(Lambda[r,:])) - np.sum(gammaln(Lambda[r,:]))
            for i in range(V):
                E_beta_lambda += (Lambda[r,i] - 1)*(digamma(Lambda[r,i]) - digamma(np.sum(Lambda[r,:])))


        Likelihood[d] = E_theta_alpha + E_z_theta + E_beta_eta + E_w_z_beta - E_theta_gamma - E_z_phi - E_beta_lambda
    return(np.sum(Likelihood))


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
    lik = D * np.log( special.gamma(K*alpha) ) 
    lik = lik - D * K * np.log(alpha)
    lik = lik + (alpha - 1) * stats
    return lik

def computeGradient(D, K, alpha, gamma, stats):
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
    # alpha_new = alpha - invHg
    
    return alpha_new


def updateAlpha(alpha, gamma, maxIter=100, tol=0.0001):
    # This function updates alpha OR eta parameter. 
    # Since the form of equations are the same, we can use this function to update eta as well.
    #
    # param1: alpha (Kx1)   OR      eta (Vx1)
    # param2: gamma (DxK)   OR   Lambda (KxV)
    # maxIter & tol: Convergence criterias. 
    isConverged = False
    
    D, K = gamma.shape
    
    epoch = 0
    bounds = []

    while(not isConverged):
        stats = computeSufficientStats(D, K, gamma)
        
        bound = computeL(D, K, alpha, stats)
        gradient = computeGradient(D, K, alpha, gamma, stats)
        hessian = computeHessian(D, K, alpha)

        #print("Epoch: %d" %epoch)
        #print("\tValue:%.5f. L: %.5f. Gradient: %.5f. Hessian: %.5f" % (alpha,bound,gradient,hessian))
        alpha = moveAlpha(alpha, gradient, hessian)
        
        #print("\tNew Value: %.5f" % (alpha))
        
        bounds.append(bound)
        epoch = epoch + 1
        
        if epoch==maxIter:
            isConverged = True
            #print("\nConverged: max iteration\n")
        else:
            if epoch>1 and np.absolute(gradient)<tol:
                isConverged = True
                #print("\nConverged: gradient close to zero at iteration: %d\n" % epoch)
        
    return alpha, bounds

def MaximizationStep(alpha, gamma, eta, Lambda):
    # Returns updated alpha and eta
    alpha, _ = updateAlpha(alpha, gamma)
    eta, _ = updateAlpha(eta, Lambda)
    
    return(alpha, eta)

def MaximizationStepUnitTest():
    K = 10
    D = 20
    V = 50
    
    alpha = np.random.rand()
    gamma = np.ones((D,K))
    eta = np.random.rand() 
    Lambda = np.ones((K,V))
    
    alpha, eta = MaximizationStep(alpha, gamma, eta, Lambda)
    
MaximizationStepUnitTest()

### End of Maximization Part ###


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
