from __future__ import division
'''
Created on 2018/5/11
@author: apple
'''

import nltk
import math 
import json
from nltk import WordPunctTokenizer
from _overlapped import NULL
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def open_file(train_path, test_path):
    f1 = open(train_path, 'r', errors = 'ignore')
    f2 = open(test_path, 'r', errors = 'ignore')
    return f1, f2

def preprocess(ftrain, ftest):
    train_data = []
    test_data = []
    word_list = []
    word_set = set()
    ftrain.readline()
    ftest.readline()
    sw = []
    f = open("D:/input/stopwords.txt")
    for line in f.readlines():
        line = line.split('\n')[0]
        sw.append(line)
    for line in ftrain.readlines():
        ls = line.split('\t')
        ls[2].lower();
        word_list = WordPunctTokenizer().tokenize(ls[2])
        train_data.append([ls[0], int(ls[1]), set(word_list)])
        for word in word_list:
            word_set.add(word)
    print("Training data processed")
    for line in ftest.readlines():
        ls = line.split('\t')
        word_list = WordPunctTokenizer().tokenize(ls[1])
        for word in word_list:
            if word in sw:
                word_list.remove(word)
        test_data.append([ls[0], 0, set(word_list)])  
    print("Testing data processed")      
    word_set = word_set - (word_set & set(sw))
    return train_data, test_data, word_set

def f_CHI(A, B, C, D):
    x = math.pow(A*D-B*C, 2)
    return x/((A+B)*(C+D));

def feature_select(train_data, word_set):

    CHI = []
    pnum = 0
    nnum = 0
    for a in train_data:
        if a[1] == 1:
            pnum += 1
        else:
            nnum += 1
    for word in word_set:
        A = B = C = D = 0
        for a in train_data:
            if word in a[2]:
                if a[1] == True:
                    A += 1
                else:
                    B += 1
        C = pnum-A
        D = nnum-B
        CHI.append([f_CHI(A, B, C, D), word])
    features = sorted(CHI, key = lambda a:a[0], reverse = True)
    json.dump(features, open('D:/input/features.txt', 'w'))
    return NULL

def feature_calculate(data, features, x, n):
    vectors = []
    for i in range(n):
        a = data[i]
        v = {}
        for b in features:
            if b[1] in a[2]:
                v[b[1]] = True
            else:
                v[b[1]] = False
        if x == True:
            vectors.append([v, a[1]])
        else:
            vectors.append(v)
    return vectors

'''
ftrain, ftest = open_file("D:/input/labeledTrainData.tsv", "D:/input/testData.tsv")
train_data, test_data, word_set = preprocess(ftrain, ftest)        
#feature_select(train_data, word_set)

features = json.load(open('D:/input/features.txt', 'r'))
features = features[:1000]
train_vectors = feature_calculate(train_data, features, True, 10000)
test_vectors = feature_calculate(test_data, features, False, 5000)
json.dump(train_vectors, open('D:/input/train_vectors.txt', 'w'))
json.dump(test_vectors, open('D:/input/test_vectors.txt', 'w'))
'''
train_vectors = json.load(open('D:/input/train_vectors.txt', 'r'))
test_vectors = json.load(open('D:/input/test_vectors.txt', 'r'))
classifier = nltk.NaiveBayesClassifier.train(train_vectors)
#classifier = nltk.classify.SklearnClassifier(LinearSVC())
#classifier.train(train_vectors)
results = classifier.classify_many([fs for fs in test_vectors])
'''
results = classifier.predict(test_vectors)
result = []
for i in range(len(results)):
    result.append(int(results[i]))
    #print(results[i])
'''
json.dump(results, open('D:/input/result.txt', 'w'))
