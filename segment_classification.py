# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import time

from pandas import ExcelFile

from pystruct.models import ChainCRF, MultiClassClf
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC

from plot_segments import plot_segments


num_segments_per_jacket = 40
add_gaussian_noise_to_features = False
sigma_noise = 0.1
plot_labeling = False
plot_coefficients = True


""" 
Load the segments and the groundtruth for all jackets
"""
path_measures = 'man_jacket_hand_measures.xls'
xl = ExcelFile(path_measures)
sheet = xl.parse(xl.sheet_names[0])
""" be careful, parse() just reads literals, does not execute formulas """
xl.close()

it = sheet.iterrows()
labels_segments = []
segments = []
for row in it:
    ide = row[1]['ide']
    segments.append(np.load(os.path.join('segments',ide+'_front.npy')))
    labels_segments.append(list(row[1].values[-num_segments_per_jacket:]))

labels_segments = np.array(labels_segments).astype(int)



""" 
Make matrices X of shape (number of jackets, number of features) 
and Y of shape (number of jackets, number of segments) where, 
for all jackets,
    X = select the features for each segment 
    Y = the grountruth label for each segment
"""
Y = labels_segments
num_jackets = labels_segments.shape[0]
num_labels = np.unique(np.ravel(labels_segments)).size

""" CHANGE THIS IF YOU CHANGE NUMBER OF FEATURES OF SEGMENTS """
num_features = 5 
X = np.zeros((num_jackets, num_segments_per_jacket, num_features))

for jacket_segments, i in zip(segments, range(num_jackets)):
    for s, j in zip(jacket_segments, range(num_segments_per_jacket)):
        """ SET WHAT THE FEATURES ARE """
        X[i,j,0:num_features] = \
           s.x0norm, s.y0norm, s.x1norm, s.y1norm, \
           s.angle/np.pi
        """ ALL POSSIBLE SEGMENT FEATURES AT PRESENT, see class Segment """
        # s.x0, s.y0, s.x1, s.y1, \
        # s.x0norm, s.y0norm, s.x1norm, s.y1norm, \
        # (s.x0norm+s.x1norm)/2., (s.y0norm+s.y1norm)/2., \
        # np.sqrt((s.x0norm-s.x1norm)**2 + (s.y0norm-s.y1norm)**2), \
        # s.angle/np.pi, \

print('X, Y done')

""" YOU CAN ADD SOME NOISE TO THE FEATURES """
if add_gaussian_noise_to_features:
    print('Noise sigma {}'.format(sigma_noise))
    X = X + np.random.normal(0.0, sigma_noise, size=X.size).reshape(np.shape(X))
      
      
model = ChainCRF(n_states=num_labels, n_features=num_features, 
                 directed=True, inference_method='max-product')
                 #'max-product','ad3', qpbo, lp

""" YOU CAN COMPARE 3 VARIANTS OF SSVM AND CHANGE THE MAIN PARAMETERS """
variant = 'FrankWolfeSSVM'
if variant=='OneSlackSSVM':
    ssvm = OneSlackSSVM(model=model, C=1.e+4, inference_cache=50, tol=0.001, \
                        verbose=0, max_iter=500)
elif variant=='NSlackSSVM':
    ssvm = NSlackSSVM(model, C=1.0e+4, tol=0.001, verbose=0, max_iter=500)
    """ NSlackSSVM needs feature normalization """
elif variant=='FrankWolfeSSVM':    
     ssvm = FrankWolfeSSVM(model=model, C=5.e+4, tol=0.001, verbose=0, max_iter=500)
else:
    assert False
     

""" 
Compare SVM with S-SVM doing k-fold cross validation, k=5, see scikit-learn.org 
"""
n_folds = 5
""" with 5 in each fold we have 4 jackets for testing, 19 for training, 
with 23 we have leave one out : 22 for training, 1 for testing"""
scores_crf = np.zeros(n_folds)
scores_svm = np.zeros(n_folds)
wrong_segments_crf = []
wrong_segments_svm = []

kf = KFold(num_jackets, n_folds=n_folds)
fold = 0
for train_index, test_index in kf: 
    print(' ')
    print('train index {}'.format(train_index))
    print('test index {}'.format(test_index))
    print('{} jackets for training, {} for testing'.\
        format(len(train_index), len(test_index)))
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]
                             
    start = time.time() 
    """ YOUR S-SVM TRAINING CODE HERE """
    ssvm.fit(X_train, Y_train)
    end = time.time()
    print('CRF learning of 1 fold has taken {} seconds'.format((end-start)/1000.0))
    
    scores_crf[fold] = ssvm.score(X_test, Y_test)
    print np.round(end - start), 'elapsed seconds to train the model'
    print("Test score with chain CRF: %f" % scores_crf[fold])
    
    """ Label the testing set and print results """
    Y_pred = ssvm.predict(X_test)
    wrong_fold_crf = np.sum(np.ravel(Y_test) - np.ravel(Y_pred) !=0)
    wrong_segments_crf.append(wrong_fold_crf)
    print('{} wrong segments out of {}'.\
        format(wrong_fold_crf, len(test_index)*num_segments_per_jacket))
    
    """ figure showing the result of classification of segments for
    each jacket in the testing part of present fold """
    if plot_labeling:
        for ti,pred in zip(test_index, Y_pred):
            print(ti)
            print(pred)
            s = segments[ti]
            plot_segments(s,caption='SSVM predictions for jacket '+str(ti+1),
                          labels_segments=pred)

    """ Train linear SVM """
    """ YOUR SVM TRAINING CODE HERE """
    svm = LinearSVC(multi_class='ovr', dual=False, C=1.e+4, tol=0.001)
    start = time.time() 
    svm.fit(np.vstack(X_train), np.hstack(Y_train))
    end = time.time()
    print('SVM training of 1 fold has taken {} seconds'.format((end-start)/1000.0))

    scores_svm[fold] = svm.score(np.vstack(X_test), np.hstack(Y_test))
    print("Test score with linear SVM: %f" % scores_svm[fold])

    Y_pred = svm.predict(np.vstack(X_test))
    wrong_fold_svm = np.sum(np.hstack(Y_test) - np.ravel(Y_pred) !=0)
    wrong_segments_svm.append(wrong_fold_svm)
    print('{} wrong segments out of {}'.\
        format(wrong_fold_svm, len(test_index)*num_segments_per_jacket))

    fold += 1
    
        
"""
Global results
"""
total_segments = num_jackets*num_segments_per_jacket
wrong_segments_crf = np.array(wrong_segments_crf)
wrong_segments_svm = np.array(wrong_segments_svm)
print(' ')
print('Scores CRF : {}'.format(scores_crf))
print('Scores SVM : {}'.format(scores_svm))
print('Wrongs CRF : {}'.format(wrong_segments_crf))
print('Wrongs SVM : {}'.format(wrong_segments_svm))
print('Final score CRF: {}, {} wrong labels in total out of {}'.\
    format(1.0 - wrong_segments_crf.sum()/float(total_segments),
           wrong_segments_crf.sum(),
           total_segments))
print('Final score SVM: {}, {} wrong labels in total out of {}'.\
    format(1.0 - wrong_segments_svm.sum()/float(total_segments),
           wrong_segments_svm.sum(),
           total_segments))


if plot_coefficients:
    name_of_labels = [
        'neck',		
        'left shoulder',
        'outer left sleeve',
        'left wrist',
        'inner left sleeve',
        'left chest',
        'waist',
        'right chest',
        'inner right sleeve',
        'right wrist',
        'outer right sleeve',
        'right shoulder',
    ]

    """ image of unary coefficients, size (num_labels, num_features)"""
    unary = ssvm.w[:num_labels*num_features]
    plt.matshow(np.reshape(unary, (num_labels, num_features)))
    plt.yticks(np.arange(num_labels), name_of_labels)
    """ CHANGE LABELS IN XTICKS ACCORDING TO YOUR FEATURE NAMES """
    plt.xticks(np.arange(num_features)+0.5, ['x0norm', 'y0norm', 'x1norm', 'y1norm', 'length', 'angle'], rotation=45)
    plt.colorbar()
    plt.show()

    """ image of pairwise coefficients, size (num_labels, num_labels)"""
    plt.matshow(ssvm.w[num_labels*num_features:].reshape(num_labels,num_labels))
    plt.xticks(np.arange(num_labels), name_of_labels, rotation=90)
    plt.yticks(np.arange(num_labels), name_of_labels)
    plt.colorbar()
    plt.show()


"""
Resultats
5 folds, posicio extrems + centre + angle/2pi, FrankWolfe
    2016, primeres 30 jaquetes 
    Wrongs CRF : [2   1  0  0  1] ->  4 de 1200 segments, 0.9966
    Wrongs SVM : [ 7 10  9  1  1] -> 28                   0.9766

    2015, 23 jaquetes
    Wrongs CRF : [0 0 0 2 0] ->  2 de 920 segments, 0.9978 
    Wrongs SVM : [7 4 3 9 2] -> 25                  0.9728
    
"""            
            
            