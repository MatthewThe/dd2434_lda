import csv
import os
import sys
import re

import numpy as np

import functions as em
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

import seaborn as sns

def main(argv):
    inputPath = argv[0]
    
    gamma, Lambda = readResults(inputPath)
    
    filename = "data_CNA_notna.txt"
    gene_labels = getGeneLabels(filename)
    
    for topic in range(len(Lambda)):
        dictionaryIdxs = Lambda[topic,:].argsort()[-10:]
        for didx in dictionaryIdxs:
            print(gene_labels[didx])
        print("")
    
    #Z = hierarchy.linkage(gamma[:100], 'single')
    #plt.figure()
    #dn = hierarchy.dendrogram(Z)
    
    gamma -= np.min(gamma)
    gamma /= gamma.sum(axis = 1)[:, np.newaxis]
    
    
    #plt.ylabel("individuals")
    #plt.xlabel("subtypes")
    g = sns.clustermap(gamma, figsize = (20,10))
    g.ax_heatmap.set_ylabel("individuals", fontsize = 24)
    g.ax_heatmap.set_xlabel("putative subtypes", fontsize = 24)
    
    plt.show()

def getGeneLabels(filename):
    filepath = "CancerData/" + filename
    status = os.path.isfile(filepath) 
    if(not status):
        zipp = zipfile.ZipFile("./data_CNA_notna.txt.zip")
        zipp.extractall("CancerData")

    with open(filepath) as infile:
        # Read header line
        first_line = infile.readline()
    
        # Read remaning lines
        gene_labels = []
        for line in infile:
            content = line.split()[0]
            gene_labels.append(content + "_neg")
            gene_labels.append(content + "_pos")
            
    return gene_labels 

def getPhi(inputPath, docIdx):
    reader = csv.reader(open(os.path.join(inputPath, 'output_phi.tsv'), 'r'), delimiter = '\t')
    phi, phiD = list(), list()
    docIdxCounter = 0
    for row in reader:
        if len(row) == 1:
            if len(phiD) > 0:
                if docIdxCounter == docIdx:
                    return np.array(phiD)
                docIdxCounter += 1
            phiD = list()
        else:  
            phiD.append(map(float, row))

def readResults(inputPath):
    reader = csv.reader(open(os.path.join(inputPath, 'output_gamma.tsv'), 'r'), delimiter = '\t')
    gamma = list()
    for row in reader:
      gamma.append(map(float, row[:-1]))
    
    reader = csv.reader(open(os.path.join(inputPath, 'output_Lambda.tsv'), 'r'), delimiter = '\t')
    Lambda = list()
    for row in reader:
      Lambda.append(map(float, row))
    
    return np.array(gamma), np.array(Lambda)

if __name__ == "__main__":
    main(sys.argv[1:])
