import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
import regex
import re
import nltk

st = stopwords.words('english')
stemmer = PorterStemmer()

def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df

def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))


word_clusters = {}

def loadWordClusters():
    infile = open('./50mpaths2.txt')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getClusterFeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)


f_path = './pdfalls.csv'
data = loadDataAsDataFrame(f_path)
texts = data['fall_description']
classes = data['fall_class']
locations = data['fall_location']
locations = [str(location) for location in locations]

classes = classes.replace('BoS','Other')

word_clusters = loadWordClusters()

training_set_size = int(0.8*len(data))
training_data = data[:training_set_size]
training_texts = texts[:training_set_size]
training_classes = classes[:training_set_size]
training_locations = locations[:training_set_size]

test_data = data[training_set_size:]
test_texts = texts[training_set_size:]
test_classes = classes[training_set_size:]
test_locations = locations[training_set_size:]


r1 = re.compile(r"(\b(lost|loss)\b)(\s*\b\w+\b){0,2}\s*(\b(balance)\b)",re.IGNORECASE)
r2 = re.compile(r"(\b(turned|turning)\b)(\s*\b\w+\b){0,2}\s*(\b(quickly|fast)\b)",re.IGNORECASE)


training_texts_preprocessed = [preprocess_text(tr) for tr in training_texts]
test_texts_preprocessed = [preprocess_text(te) for te in test_texts]

training_texts_preprocessed_locations = [preprocess_text(tr) for tr in training_locations]
test_texts_preprocessed_locations = [preprocess_text(te) for te in test_locations]

training_texts_preprocessed_balance_check = [[len(re.findall(r1, tr))] for tr in training_texts]
test_texts_preprocessed_balance_check = [[len(re.findall(r1, tr))] for tr in test_texts]

training_texts_preprocessed_turning_check = [[len(re.findall(r2, tr))] for tr in training_texts]
test_texts_preprocessed_turning_check = [[len(re.findall(r2, tr))] for tr in test_texts]

training_texts_preprocessed_clusters = [getClusterFeatures(tr) for tr in training_texts]
test_texts_preprocessed_clusters = [getClusterFeatures(tr) for tr in test_texts]

vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=200)

location_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=200)

cluster_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                 max_features=200)

training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed).toarray()
training_data_vectors = [np.append(training_data_vectors[i],len(training_texts_preprocessed[i])) for i in range(0,len(training_data_vectors))]
training_data_vectors_locations = location_vectorizer.fit_transform(training_texts_preprocessed_locations).toarray()
training_data_vectors_clusters = cluster_vectorizer.fit_transform(training_texts_preprocessed_clusters).toarray()

test_data_vectors = vectorizer.transform(test_texts_preprocessed).toarray()
test_data_vectors = [np.append(test_data_vectors[i],len(test_texts_preprocessed[i])) for i in range(0,len(test_data_vectors))]
test_data_vectors_locations = location_vectorizer.transform(test_texts_preprocessed_locations).toarray()
test_data_vectors_clusters = cluster_vectorizer.transform(test_texts_preprocessed_clusters).toarray()

training_data_vectors = np.concatenate((training_data_vectors, training_data_vectors_locations), axis=1)
test_data_vectors = np.concatenate((test_data_vectors,test_data_vectors_locations),axis=1)

training_data_vectors = np.concatenate((training_data_vectors, training_texts_preprocessed_balance_check), axis=1)
test_data_vectors = np.concatenate((test_data_vectors,test_texts_preprocessed_balance_check),axis=1)

training_data_vectors = np.concatenate((training_data_vectors, training_texts_preprocessed_turning_check), axis=1)
test_data_vectors = np.concatenate((test_data_vectors,test_texts_preprocessed_turning_check),axis=1)

training_data_vectors = np.concatenate((training_data_vectors, training_data_vectors_clusters), axis=1)
test_data_vectors = np.concatenate((test_data_vectors,test_data_vectors_clusters),axis=1)


def grid_search_hyperparam_space(params, pipeline, folds, training_data_vectors, training_classes):#folds, x_train, y_train, x_validation, y_validation):
    grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='accuracy',n_jobs=-1)
    grid_search.fit(training_data_vectors, training_classes)
    
    print('Best hyperparameters:')
    print(grid_search.best_params_)


    #CLASSIFY AND EVALUATE 
    predictions = grid_search.predict(test_data_vectors)
    print('Performance on held-out test set ... :')

    print(accuracy_score(predictions,test_classes))
    print(f1_score(test_classes, predictions, average='micro', labels=['CoM']))
    print(f1_score(test_classes, predictions, average='macro'))
    
    return grid_search


svm_classifier = svm.SVC(gamma='scale',verbose=True)
pipeline = Pipeline(steps = [('svm_classifier',svm_classifier)])

grid_params = {
     'svm_classifier__C': [1, 4, 16, 64],
     'svm_classifier__kernel': ['linear','rbf']
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


randomForest_classifier = RandomForestClassifier(random_state=1)
pipeline = Pipeline(steps = [('randomForest_classifier',randomForest_classifier)])

grid_params = {
     'randomForest_classifier__n_estimators': [10, 50, 100, 200],
     'randomForest_classifier__max_depth': [None,2,5,10]
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


gnb_classifier = GaussianNB()
pipeline = Pipeline(steps = [('gnb',gnb_classifier)])

grid_params = {
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(random_state=0)
pipeline = Pipeline(steps = [('gpc',gpc)])

grid_params = {
     'gpc__kernel': [1.0 * RBF(1.0), 2.0 * RBF(2.0), 2.0 * RBF(5.0), 5.0 * RBF(5.0)]
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


ridge_classifier = RidgeClassifier()
pipeline = Pipeline(steps = [('ridge_classifier',ridge_classifier)])

grid_params = {
     'ridge_classifier__alpha': [1.0, 2.0, 5.0, 10.0]
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


dtc = DecisionTreeClassifier(random_state=0)
pipeline = Pipeline(steps = [('dtc',dtc)])

grid_params = {
     'dtc__criterion': ["gini","entropy"],
     'dtc__splitter': ["best","random"],
     'dtc__max_depth': [None,2,5,10]
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


eclf = VotingClassifier(estimators=[('svc', svm_classifier), ('rf', randomForest_classifier), ('gnb', gnb_classifier)])
pipeline = Pipeline(steps = [('eclf',eclf)])

grid_params = {
}

folds = 10
grid = grid_search_hyperparam_space(grid_params,pipeline,folds,training_data_vectors,training_classes)


import matplotlib.pyplot as plt
performance_dict = {}

#### Values were generated using the code above and then hard coded on a jupyter notebook to generate the graph ##########

performance_dict[0.2] = {'accuracy':0.6559139784946236,'micro':0.7866666666666665,'macro':0.4488888888888888}
performance_dict[0.3] = {'accuracy':0.6097560975609756,'micro':0.7499999999999999,'macro':0.43055555555555547}
performance_dict[0.4] = {'accuracy':0.5714285714285714,'micro':0.7115384615384616,'macro':0.4391025641025641}
performance_dict[0.5] = {'accuracy':0.5,'micro':0.6419753086419753,'macro':0.4067019400352734}
performance_dict[0.6] = {'accuracy':0.5319148936170213,'micro':0.6562500000000001,'macro':0.4614583333333334}
performance_dict[0.7] = {'accuracy':0.5714285714285714,'micro':0.6808510638297872,'macro':0.5143385753931545}
performance_dict[0.8] = {'accuracy':0.5833333333333334,'micro':0.6428571428571429,'macro':0.5714285714285714}
performance_dict[0.9] = {'accuracy':0.8333333333333334,'micro':0.8571428571428571,'macro':0.8285714285714285}

accuracy_list = [performance_dict[size]['accuracy'] for size in performance_dict]
micro_list = [performance_dict[size]['micro'] for size in performance_dict]
macro_list = [performance_dict[size]['macro'] for size in performance_dict]
size_list = [20,30,40,50,60,70,80,90]

plt.plot(size_list,accuracy_list,label='Accuracy')
plt.plot(size_list,micro_list,label='Micro F1')
plt.plot(size_list,macro_list,label='Macro F1')
plt.xlabel('Training set size (percentage)')
plt.ylabel('Ridge Classifier Performance')
plt.legend()
plt.show()



