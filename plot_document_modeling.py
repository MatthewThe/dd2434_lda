import csv
import os
import sys
import re

import numpy as np

import functions as em

def main(argv):
    inputPath = argv[0]
    
    gamma, Lambda = readResults(inputPath)
    
    tf, labels_1, labels_2, topic_texts, dictionary, data = em.loadData('reutersdata', 'earn', 'grain', maxDocs = len(gamma), returnData = True)
    
    samples = 0
    minWordsFromTopic = 5
    numTopics = 4
    colors = ['red','orange','green','blue']
    gamma = gamma - np.min(gamma)
    for docIdx, row in enumerate(gamma):
        if np.sum(row >= minWordsFromTopic) == numTopics:
            topics = np.where(row >= minWordsFromTopic)
            if len(data[docIdx].split()) < 100:
                print(data[docIdx])
                wd = em.wordCountToFlatArray(tf[docIdx,:])
                
                phiAssigned = np.where(getPhi(inputPath, docIdx) > 0.9)
                print("")
                
                topicToColor = dict()
                for topicIdx, topic in enumerate(topics[0]):
                    dictionaryIdxs = Lambda[topic,:].argsort()[-10:]
                    print("Num words from this topic: %f" % row[topic])
                    for didx in dictionaryIdxs:
                        print(dictionary[didx])
                    print("")
                    topicToColor[topic] = colors[topicIdx]
                
                wordToColoredWord = dict()
                for wordIdx, topicIdx in zip(*phiAssigned):
                    if topicIdx in topicToColor:
                        wordToColoredWord[dictionary[wd[wordIdx]]] = topicToColor[topicIdx]
                
                newText = data[docIdx]
                for word, color in wordToColoredWord.iteritems():
                    newText = re.sub(r'\s(%s)\s' % word ,r' \\textcolor{%s}{\1} ' % color, newText, flags=re.I)
                
                print(newText)
                print("")
                samples += 1
                if samples > 10:
                    break

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
