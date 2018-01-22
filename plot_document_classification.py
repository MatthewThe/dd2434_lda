import sys
import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import functions as em

def main(argv):
    gamma = readResults(argv[0])
    
    tf, labels_1, labels_2, topic_texts, dictionary = em.loadData('reutersdata', 'earn', 'grain', maxDocs = 8000)
    
    mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim = DocumentClassification(gamma, tf, labels_1)
    plt.figure()
    plotSVMModelAccuracy(mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim)
    
    mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim = DocumentClassification(gamma, tf, labels_2)
    plt.figure()
    plotSVMModelAccuracy(mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim)
    
    plt.show()

def readResults(inputPath):
    reader = csv.reader(open(os.path.join(inputPath, 'output_gamma.tsv'), 'r'), delimiter = '\t')
    gamma = list()
    for row in reader:
        gamma.append(map(float, row[:-1]))
    return np.array(gamma)

# Replace with gamma for topic distribution
def MockTopicData():
    gamma_matrix=np.random.rand(8000, 50)

    return (gamma_matrix)

#Classify documents into binary class
# Run for EARN vs NOT EARN and GRAIN vs NOT GRAIN, recursively
def DocumentClassification(gamma_matrix, tf, labels_1_array, num_runs = 3):
    print("*** DOCUMENT CLASSIFICATION ***")
    
    # initialize the svm parameters for grid search and run both linear and radial basis function as kernels
    #parameters = {'kernel': ('linear','rbf'), 'C': [1,3,5,7,10], 'gamma':[0.01,0.05,0.10,0.3,0.5]}
    parameters = {'C': [1,3,5,7,10]}
    
    # proportion of test data
    train_data_size = [0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]
    
    print("*** LDA gamma features ***")
    ## building classifier for low dimension data D(8000) X K(50) (for LDA features)
    mean_stats_low_dim, sd_stats_low_dim = trainSVMs(num_runs, gamma_matrix, labels_1_array, parameters, train_data_size)
    
    if True:
      print("*** Word features ***")
      ### building a classifier for high dimension data D(8000) X w (for word feature)
      mean_stats_high_dim, sd_stats_high_dim = trainSVMs(num_runs, tf, labels_1_array, parameters, train_data_size)
    else:
      mean_stats_high_dim, sd_stats_high_dim = np.nan, np.nan

    return (mean_stats_low_dim, sd_stats_low_dim, mean_stats_high_dim, sd_stats_high_dim)
    
def trainSVMs(num_runs, data_array, labels_1_array, parameters, train_data_size):
    low_dim_acc = []
    for k in range(num_runs):
        print("*** Run %d ***" % k)
        acc_list_low_dim = []
        for i in train_data_size:
            print("*** test data size %f ***" % i)
            X_topic_train, X_topic_test, y_topic_train, y_topic_test = train_test_split(data_array, labels_1_array, test_size = 1.0 - i, random_state = k)
            svr = svm.LinearSVC()
            grid = GridSearchCV(svr, parameters)
            grid.fit(X_topic_train, y_topic_train)
            predicted = grid.predict(X_topic_test)
            acc_list_low_dim.append(accuracy_score(y_topic_test, predicted))
        low_dim_acc.append(acc_list_low_dim)
        print acc_list_low_dim
    accuracy_low_dim = np.array(low_dim_acc)
    mean_stats_low_dim = np.mean(accuracy_low_dim, axis=0)
    sd_stats_low_dim = np.std(accuracy_low_dim, axis=0)
    
    return mean_stats_low_dim, sd_stats_low_dim
    #print (sd_stats_low_dim)
    
# Training data - accuracy plot (fig. 10 in Blei et al.)
def plotSVMModelAccuracy(mean_low_dim, sd_low_dim, mean_high_dim, sd_high_dim):
    #scatter plot for LDA features
    plt.errorbar(np.array([0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]), mean_low_dim,
                 yerr = sd_low_dim, label="LDA features", fmt="s--", linewidth=1)
    #scatter plot for word features
    plt.errorbar(np.array([0.01 ,0.05 ,0.1, 0.15, 0.2, 0.25]), mean_high_dim, yerr= sd_high_dim, label="Word features", fmt="s-", linewidth=1)
    plt.rc('axes', labelsize = 15)
    plt.ylim(0.7, 1.0)
    plt.xlim(0,0.3)
    plt.xlabel('Proportion of data used for training')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right', shadow=True, fontsize='x-large', prop={'size': 10})

if __name__ == "__main__":
    main(sys.argv[1:])
