import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma 

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


### Start of Maximization Part ###

#def updateAlpha(D, K, alpha, gamma):
#    # Newton-Rhaphson algorithm for updating alpha
#    alpha = np.random.rand(D,1)
#    return(alpha)
    
#def updateEta(V, K, Lambda, eta): 
#     # Newton-Rhaphson algorithm for updating alpha
#    eta = np.random.rand(V, 1)
#    return(eta)

# TODO: Check alpha, readjust.
def calculateLowerBound(param1, param2):
    # Given a parameter pair (alpha&gamma or eta&lambda), this function calculates the lower bound (Appendix A.4.2)
    d2, d1 = param2.shape
    
    bound = 0
    sum_param1 = np.sum(param1)
    bound = bound + d2 * np.log(polygamma(0, sum_param1)) # gamma function
    
    temp = 0
    for i in range(d1):
        temp = temp + np.log(polygamma(0, param1[i])) # gamma function
    bound = bound - d2 * temp
    
    sum_param2 = np.sum(param2, axis=1)
    di_sum_param2 = polygamma(1, sum_param2) # digamma function
    
    temp = 0
    for d in range(d2):
        for i in range(d1):
            diff = polygamma(1, param2[d,i]) - di_sum_param2[d]  # digamma function
            temp = temp + (param1[i]-1) * diff
    bound = bound + temp
    
    return bound

def calculateGradient(param1, param2):
    # Given a parameter pair (alpha&gamma or eta&lambda), this function calculates the gradient dL (Appendix A.4.2)
    d2, d1 = param2.shape
    
    gradient = np.zeros((d1))
    
    sum_param1 = np.sum(param1)
    gradient = gradient + d2 * polygamma(1, sum_param1) # digamma function
    
    sum_param2 = np.sum(param2, axis=1)
    di_sum_param2 = polygamma(1, sum_param2) # digamma function
    gradient = gradient - np.sum(di_sum_param2)
    
    for i in range(d1):
        temp = - d2 * polygamma(1, param1[i]) # digamma function
        for d in range(d2):
            temp = temp + polygamma(1, param2[d,i])  # digamma function
        
        gradient[i] = gradient[i] + temp
    
    return gradient

def calculateHessian(param1, d2):
    # Given a parameter (alpha or eta), this function calculates the hessian d2L (Appendix A.4.2 and A.2)
    sum_param1 = np.sum(param1)
    z = - polygamma(2, sum_param1) # trigamma function
    hessian_diag = d2 * polygamma(2, param1) # trigamma function
    hessian = np.diag(hessian_diag) + z
    
    return hessian_diag, z, hessian

def calculateC(gradient, hessian_diag, z):
    # Given gradient and hessian, this function calculates C value (Appendix A.2)
    d1 = gradient.shape[0]
    
    up = np.divide(gradient, hessian_diag)
    up = np.sum(up)
    bottom = 1 / z
    for j in range(d1):
        bottom = bottom + (1 / hessian_diag[j])

    c = up / bottom
    return c

def calculateInvHg(gradient, hessian_diag, c):
    # Given gradient, hessian and c; this function calculates inv_H*g (i.e the step size) (Appendix A.2)
    invHg = np.divide((gradient - c), hessian_diag)
    return invHg

def moveParam(param1, invHg):
    # This function updates the parameter (p_new = p_old - inv_H(p_old)*g(p_old)) (Appendix A.2)
    param1_new = param1 - invHg
    return param1_new

#def updateAlpha(D, K, alpha, gamma):
#    # Newton-Rhapson algorithm for updating alpha
#    return(alpha)
def updateParam(param1, param2, maxIter=100, tol=0.001, isDisplay=False):
    # This function is the generalized version of updating alpha and eta parameters.
    #
    # param1  (d1 x 1): alpha (Kx1)   OR      eta (Vx1)
    # param2 (d2 x d1): gamma (DxK)   OR   Lambda (KxV)
    # maxIter & tol: Convergence criterias. 
    # isDisplay: Printing values for debugging purpose
    isConverged = False
    
    d2,d1 = param2.shape
    epoch = 0
    bounds = []
    
    prev_gradient = np.zeros((d1))
    
    while(not isConverged):
        bound = calculateLowerBound(param1, param2)
        gradient = calculateGradient(param1, param2)
        hessian_diag, z, hessian = calculateHessian(param1, d2)
        c = calculateC(gradient, hessian_diag, z)
        invHg = calculateInvHg(gradient, hessian_diag, c)
        param1_new = moveParam(param1, invHg)
        
        if(isDisplay):
            print("\n\tIteration: %d" %e)
            print("Bound")
            print(bound)
            print("\nGradient")
            print(gradient)
            print("\nHessian Parts")
            print(hessian_diag)
            print(z)
            print("\nHessian")
            print(hessian)
            print("\nC")
            print(c)
            print("\ninvHg")
            print(invHg)
            print("\nPrev Param:")
            print(param1)
            print("\nNew Param:")
            print(param1_new)

        param1 = param1_new
        bounds.append(bound)
        epoch = epoch + 1
        
        if epoch==100:
            isConverged = True
            print("\nConverged: max iteration\n")
        else:
            if epoch>1:
                diff = np.isclose(gradient, prev_gradient)
                if np.all(diff):
                    isConverged = True
                    print("\nConverged: gradient diff at iteration: %d\n" % epoch)
            prev_gradient = gradient
    return param1, bounds

#def MaximizationStep(D, V, K,  alpha, gamma, phi, Lambda, eta, likelihood):
#    # Returns updated alpha, eta
#    alpha = updateAlpha(D, K, alpha, gamma)    
#    eta = updateEta(V, K, Lambda, eta)
#    return(alpha, eta)
def MaximizationStep(alpha, gamma, eta, Lambda):
    # Returns updated alpha and eta
    alpha, _ = updateParam(alpha, gamma)
    eta, _ = updateParam(eta, Lambda)
   
    return(alpha, eta)

def MaximizationStepUnitTest():
    K = 10
    D = 100
    V = 50
    
    alpha = np.random.rand(K) + 1.5
    gamma = np.ones((D,K))
    eta = np.random.rand(V) + 1.5
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
