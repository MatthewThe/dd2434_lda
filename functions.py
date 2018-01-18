import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, polygamma, gammaln 
from scipy import special
from sklearn.feature_extraction.text import CountVectorizer
import os
from bs4 import BeautifulSoup
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

np.random.seed(1)


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
    alpha = createAlpha(maxEta)
    eta = createEta(maxEta)
    beta = createBeta(eta, V, K)
    theta = createTheta(alpha, K, D)
    Z = createZ(theta, D, N, K)
    w, w_counts = createW(beta, Z, D, N, V)
    
    return alpha, eta, beta, theta, Z, w, w_counts
##### End of Synthetic Data Generation #####

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
            soup = (filestring)
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
#tf = loadMockData(10, 10)

D = 10
N = 20
K = 2
V = 10
alpha_original, eta_original, _, theta_original, _, _, tf = createSample(D,N,K,V,maxAlpha=5, maxEta=10)
print("*** Original alpha: %.5f, Original eta: %.5f ***\n" % (alpha_original, eta_original))
print("*** Original theta ***")
print(theta_original)
# Initialize parameters
    
# tf: input_data
K = 10 # Number of Topics
D = getDataDimensions(tf)[0] # Number of Documents
V = getDataDimensions(tf)[1] # Number of words in dictionary
N = np.max(np.sum(tf, axis = 1))
beta_init = np.zeros((D, N)) # beta
alpha_init = [50.0/K] * K # alpha
gamma_init = [alpha_init[0] + float(N)/K] * K # gamma
phi_init = np.full((K, N), 1./K) # phi
eta = 0.001 # eta
Lambda_init = np.zeros((D, N)) # lambda
loglikelihoodVEM = 0 # Likelihood to be minimized



## Functions

def ComputeLikelihood(tf, alpha, beta, gamma, phi, Lambda):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the betas. (smoothed version) 

    Likelihood = np.zeros((D,1))
    for d in range(D):
        #Hazal's notes, eq. 6
        E_theta_alpha = 0
        E_theta_alpha = gammaln(alpha*K) \
                        - K * (gammaln(alpha)) \
                        + (alpha-1) * computeSufficientStats(D, K, gamma)

        #Hazal's notes, eq. 7
        E_z_theta = 0
        wd = np.repeat(range(V), tf[d,:])
        for n in range(len(wd)):
            for r in range(K):
               E_z_theta += phi[d,n,r] *(digamma(gamma[d,r]) - digamma(np.sum(gamma[d,:])))

        #Hazal's notes, eq. 8
        E_beta_eta = 0
        for r in range(K):
            E_beta_eta += gammaln(eta*V) - V*gammaln(eta)
            for i in range(V):
                 E_beta_eta += (eta - 1)*(digamma(Lambda[r,i]) - digamma(np.sum(Lambda[r,:])))

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

def ComputeDocumentLikelihood(tf, alpha, beta, gamma, phi, Lambda, d):
    # Computes likelihood of the variational model (eq. 15).
    # Lambda is the additional hyperparameter for the betas. (smoothed version) 

    Likelihood = np.zeros((D,1))
    #Hazal's notes, eq. 6
    E_theta_alpha = 0
    E_theta_alpha = gammaln(np.sum(alpha*K)) \
                    - (gammaln(alpha)*K) \
                    + np.sum([((alpha - 1) \
                        *(digamma(gamma[d, i]) - digamma(np.sum(gamma[d,:])))) \
                       for i in range(K)])

    #Hazal's notes, eq. 7
    E_z_theta = 0
    wd = np.repeat(range(V), tf)
    for n in range(len(wd)):
        for r in range(K):
           E_z_theta += phi[d,n,r] *(digamma(gamma[d,r]) - digamma(np.sum(gamma[d,:])))

    #Hazal's notes, eq. 8
    E_beta_eta = 0
    for r in range(K):
        E_beta_eta += gammaln(eta*V) - V*gammaln(eta)
        for i in range(V):
            E_beta_eta += (eta - 1)*(digamma(Lambda[r,i]) - digamma(np.sum(Lambda[r,:])))

    #Hazal's notes, eq. 9
    E_w_z_beta = 0
    wd = np.repeat(range(V), tf)
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
    wd = np.repeat(range(V), tf)
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
    return(Likelihood[d])


def ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 10**(-4), EstepMaxIterations = 100):
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

    iterations = 0
    likelihood = 10**(-10) 
    converged = False
    while(not(converged)):
        if(iterations > EstepMaxIterations):
            raise ValueError('E-step not converged after %d iterations' %iterations)
        
        iterations += 1
        phi, gamma = ExpectationPhiGamma(tf, K, D, N, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 10**(-5), EstepMaxIterations = 10)
        Lambda[:,:] = eta
        for d in range(D):
            wd = np.repeat(range(V), tf[d,:])
            for k in range(K):
                for n in range(len(wd)):        
                    Lambda[k,wd[n]] += phi[d,n,k]
        
        newLikelihood = ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda)
        dlikelihood = abs((newLikelihood - likelihood)/likelihood)
        print(newLikelihood)
        likelihood = newLikelihood
        if(dlikelihood < EstepConvergeThreshold):
            print('E-step converged after %d iterations' %iterations)
            converged = True
    return(phi, gamma, Lambda, likelihood)


def ExpectationPhiGamma(tf, K, D, N, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 10**(-5), EstepMaxIterations = 10):
    for d in range(D):
        likelihood = 10**(-10) 
        converged = False
        iterations = 0
        while(not converged):
            iterations += 1
            wd = np.repeat(range(V), tf[d,:])
            for n in range(len(wd)):
                for k in range(K):
                    phi[d,n,k] = np.exp(digamma(gamma[d,k]) - digamma(np.sum(gamma[d,:])) + digamma(Lambda[k,wd[n]]) - digamma(np.sum(Lambda[k,:])))
                phi[d,n,:] /= sum(phi[d,n,:])
            #for d in range(D):
            gamma[d,:] = alpha + np.sum(phi[d,:,:], axis = 0)
            newLikelihood = ComputeDocumentLikelihood(tf[d], alpha, eta, gamma, phi, Lambda, d)
            #newLikelihood = 0.1
            dlikelihood = abs((newLikelihood - likelihood)/likelihood)
            likelihood = newLikelihood
            #print(likelihood)
            if(dlikelihood < EstepConvergeThreshold):
                print('E-step phi gamma converged after %d iterations' %iterations)
                print(likelihood)
                converged = True
    
    return(phi, gamma)   

def ExpectationStepUnitTest():
    alpha = np.random.rand()
    eta = np.random.rand()
    gamma = np.random.rand(D,K)
    phi = np.random.rand(D,N,K)
    Lambda = np.random.rand(K,V)
    
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)
    phi, gamma, Lambda, likelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)

#ExpectationStepUnitTest()


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
    
    #log_alpha_new = np.log(alpha) - invHg
    #alpha_new = np.exp(log_alpha_new)
    
    # OR!!
    alpha_new = alpha - invHg
    
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
            print("\nConverged: max iteration\n")
        else:
            if epoch>1 and np.absolute(gradient)<tol:
                isConverged = True
                print("\nConverged: gradient close to zero at iteration: %d\n" % epoch)
        
    return alpha, bounds

def MaximizationStep(alpha, gamma, eta, Lambda):
    # Returns updated alpha and eta
    alpha, _ = updateAlpha(alpha, gamma)
    eta, _ = updateAlpha(eta, Lambda)
    print("alpha: %.10f", alpha)
    print("eta: %.10f", eta)
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


def VariationalExpectationMaximization(tf, K, D, N, alpha, eta, gamma, phi, Lambda, EstepConvergeThreshold = 10**(-5), EstepMaxIterations = 10):
    # Calculates variational parameters gamma, phi, and lambda iteratively until convergence
    likelihood = 10**(-10) 
    converged = False
    iterations = 0
    while(not converged):
        iterations += 1
        phi, gamma, Lambda, newLikelihood = ExpectationStep(tf, K, D, N, alpha, eta, gamma, phi, Lambda)
        print(newLikelihood)
        alpha, eta = MaximizationStep(alpha, gamma, eta, Lambda)
        
        dlikelihood = abs((newLikelihood - likelihood)/likelihood)
        likelihood = newLikelihood
        print(ComputeLikelihood(tf, alpha, eta, gamma, phi, Lambda))
        print("gamma")
        print(gamma)
        if(dlikelihood < EstepConvergeThreshold):
            print('EM-step converged after %d iterations' %iterations)
            converged = True
    return(gamma, Lambda, phi)
    
def VariationalExpectationMaximizationUnitTest():
    alpha = np.random.rand()
    gamma = np.random.rand(D,K)
    eta = np.random.rand() 
    Lambda = np.random.rand(K,V)
    phi = np.random.rand(D,N,K)
    for d in range(D):
      for n in range(N):
        phi[d,n,:] /= np.sum(phi[d,n,:])
    
    VariationalExpectationMaximization(tf, K, D, N, alpha, eta, gamma, phi, Lambda)
    
  
#VariationalExpectationMaximizationUnitTest()  
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
def loadData(folderName, keyword_1, keyword_2):
    data = []
    labels_1 = []
    topic_of_interest_1 = keyword_1
    labels_2 = []
    topic_of_interest_2 = keyword_2
    for file in os.listdir(folderName):
        if file.endswith('.sgm'):
            f = open(folderName + '/' + file, 'rb')
            # print('reutersdata/' + file)
            filestring = f.read()
            soup = BeautifulSoup(filestring)
            contents = soup.findAll('text')
            for content in contents:
                data.append(content.text)
            topics = soup.findAll('topics')
            for topic in topics:
                labels_1.append(topic_of_interest_1 in topic.text)
                labels_2.append(topic_of_interest_2 in topic.text)
        if len(data) > 7000:
            break
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(data)

    return (tf, labels_1, labels_2)

#tf, labels_1, labels_2 = loadData('reutersdata', 'earn', 'grain')

# Replace with gamma for topic distribution
def MockTopicData():
    gamma_matrix=np.random.rand(8000, 50)

    return (gamma_matrix)

### !!! make sure you have the same documents and labels for comparison of classifications !!!!
#gamma_matrix = MockTopicData()

# Call labels labels_1 earn, labels_2 grain
#labels_1_array = np.asarray(labels_1)
#labels_2_array = np.asarray(labels_2)


#Classify documents into binary class
# Run for EARN vs NOT EARN and GRAIN vs NOT GRAIN, recursively
def DocumentClassification():

    # initialize the svm parameters for grid search and run both linear and radial basis function as kernels
    parameters = {'kernel': ('linear','rbf'), 'C': [1,3,5,7,10], 'gamma':[0.01,0.05,0.10,0.3,0.5]}

    # proportion of test data
    test_data_size = [0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]

    ## building classifier for low dimension data D(8000) X K(50) (for LDA features)
    low_dim_acc = []
    for k in range(0,6):
        acc_list_low_dim = []
        for i in test_data_size:
            X_topic_train, X_topic_test, y_topic_train, y_topic_test = train_test_split(gamma_matrix, labels_1_array, test_size = i, random_state = k)
            svr = svm.SVC()
            grid = GridSearchCV(svr, parameters)
            grid.fit(X_topic_train, y_topic_train)
            predicted = grid.predict(X_topic_test)
            acc_list_low_dim.append(accuracy_score(y_topic_test, predicted))
        low_dim_acc.append(acc_list_low_dim)
    accuracy_low_dim = np.array(low_dim_acc) * 100
    mean_stats_low_dim = np.mean(accuracy_low_dim, axis=0)
    sd_stats_low_dim = np.std(accuracy_low_dim, axis=0)
    #print (sd_stats_low_dim)


    ### building a classifier for high dimension data D(8000) X w (for word feature)
    high_dim_acc = []
    for k in range(0,6):
        acc_list_high_dim = []
        for i in test_data_size:
            X_word_train, X_word_test, y_word_train, y_word_test = train_test_split(tf, labels_1_array, test_size = i, random_state = k)
            svr = svm.SVC()
            grid = GridSearchCV(svr, parameters)
            grid.fit(X_word_train, y_word_train)
            predicted = grid.predict(X_word_test)
            acc_list_high_dim.append(accuracy_score(y_word_test, predicted))
        high_dim_acc.append(acc_list_high_dim)
    accuracy_high_dim = np.array(high_dim_acc) * 100
    mean_stats_high_dim = np.mean(accuracy_high_dim, axis=0)
    sd_stats_high_dim = np.std(accuracy_high_dim, axis=0)
    #print (sd_stats_high_dim)

    return (mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim)

# Get statistics (mean and standard deviation) for plotting
#mean_low_dim, sd_low_dim,  mean_high_dim, sd_high_dim = DocumentClassification()

# Training data - accuracy plot (fig. 10)
def plotSVMModelAccuracy():
    #scatter plot for LDA features
    plt.errorbar(np.array([0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]), mean_low_dim,
                 yerr = sd_low_dim, label="LDA features", fmt="s--", linewidth=1)
    #scatter plot for word features
    plt.errorbar(np.array([0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]), mean_high_dim, yerr= sd_high_dim, label="Word features", fmt="s-", linewidth=1)
    plt.rc('axes', labelsize = 15)
    plt.ylim(70, 100)
    plt.xlim(0,0.3)
    plt.xlabel('Proportion of data used for training')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', shadow=True, fontsize='x-large', prop={'size': 10})
    plt.show()

#plotSVMModelAccuracy()

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
