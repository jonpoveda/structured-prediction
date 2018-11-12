# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
import pandas

import os
import matplotlib.pyplot as plt
import numpy as np

from pandas import ExcelFile

from pystruct.models import ChainCRF, MultiClassClf
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from typing import Tuple, Union, Sequence

from plot_segments import plot_segments


def load_sheet(path: Union[str, Path]) -> pandas.DataFrame:
    """ Load the segments and the groundtruth for all jackets """
    path_measures = 'man_jacket_hand_measures.xls'
    with ExcelFile(path_measures) as xl:
        # be careful, parse() just reads literals, does not execute formulas
        sheet = xl.parse(xl.sheet_names[0])
    return sheet


def load_segments(sheet: pandas.DataFrame,
                  num_segments_per_jacket: int,
                  ) -> Tuple[list, np.ndarray]:

    it = sheet.iterrows()
    labels_segments = []
    segments = []

    for row in it:
        ide = row[1]['ide']
        segments.append(np.load(os.path.join('segments', ide + '_front.npy'),
                                encoding='latin1'))
        labels_segments.append(list(row[1].values[-num_segments_per_jacket:]))

    labels_segments = np.array(labels_segments).astype(int)

    return segments, labels_segments


def show_groundtruth(n: int,
                     sheet: pandas.DataFrame,
                     segments: Sequence,
                     labels_segments: np.ndarray
                     ) -> None:
    """ Show groundtruth for the n-jacket """
    plot_segments(segments[n], sheet.ide[n], labels_segments[n])
    plt.show(block=True)


def prepare_data(segments,
                 labels_segments,
                 num_segments_per_jacket,
                 num_features,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make matrices X of shape (number of jackets, number of features)
    and Y of shape (number of jackets, number of segments) where,
    for all jackets,
        X = select the features for each segment
        Y = the grountruth label for each segment

    X contains segments and is of shape [n_samples, n_segments, segment_size]

    Y contains labels and is of shape [n_samples, n_segments]

    Returns:
        segments, labels
    """
    Y = labels_segments
    num_jackets = labels_segments.shape[0]

    X = np.zeros((num_jackets, num_segments_per_jacket, num_features))

    for jacket_segments, i in zip(segments, range(num_jackets)):
        for s, j in zip(jacket_segments, range(num_segments_per_jacket)):
            """ set the features """
            X[i, j, 0:num_features] = \
                s.x0norm, s.y0norm, s.x1norm, s.y1norm, \
                (s.x0norm + s.x1norm) / 2., (s.y0norm + s.y1norm) / 2., \
                s.angle / (2 * np.pi)
            """ all possible features at present, see segment.py """
            # s.x0, s.y0, s.x1, s.y1, \
            # s.x0norm, s.y0norm, s.x1norm, s.y1norm, \
            # (s.x0norm+s.x1norm)/2., (s.y0norm+s.y1norm)/2., \
            # np.sqrt((s.x0norm-s.x1norm)**2 + (s.y0norm-s.y1norm)**2), \
            # s.angle, \

    return X, Y


def add_gaussian_noise(features: np.ndarray, sigma_noise: float) -> np.ndarray:
    """ Add some gaussian noise to the features """
    print(f'Noise sigma {sigma_noise}')
    noise = np.random.normal(0.0, sigma_noise, size=features.size)
    return features + noise.reshape(np.shape(features))


def compare_svm_and_ssvm(X: np.ndarray,
                         Y: np.ndarray,
                         svm: LinearSVC,
                         ssvm: FrankWolfeSSVM,
                         n_folds: int,
                         segments: list,
                         show_labeling: bool = False,
                         ) -> Tuple[np.ndarray, ...]:
    """ Compare SVM with S-SVM doing k-fold cross validation

    Returns:
        scores_svm, wrong_segments_svm, scores_crf, wrong_segments_crf
    """
    scores_crf = np.zeros(n_folds)
    scores_svm = np.zeros(n_folds)
    wrong_segments_crf = []
    wrong_segments_svm = []

    kf = KFold(n_splits=n_folds)

    fold = 0
    print(f'Using k={n_folds} k-fold training')
    for train_index, test_index in kf.split(X):
        # print(' ')
        # print('train index {}'.format(train_index))
        # print('test index {}'.format(test_index))
        print(f'{fold+1}-fold: '
              f'{len(train_index)} jackets for training, '
              f'{len(test_index)} for testing')
        X_train: np.ndarray = X[train_index]
        Y_train: np.ndarray = Y[train_index]
        X_test: np.ndarray = X[test_index]
        Y_test: np.ndarray = Y[test_index]

        """ YOUR S-SVM TRAINING CODE HERE """
        ssvm.fit(X_test, Y_test)

        """ LABEL THE TESTING SET AND PRINT RESULTS """
        scores_crf[fold] = ssvm.score(X_test, Y_test)
        Y_pred = ssvm.predict(X_test)
        wrong_segments_crf.append(np.sum(Y_test != Y_pred))

        """ figure showing the result of classification of segments for
        each jacket in the testing part of present fold """
        if plot_labeling:
            for ti, pred in zip(test_index, Y_pred):
                print(ti)
                print(pred)
                s = segments[ti]
                plot_segments(
                    s,
                    caption='SSVM predictions for jacket ' + str(ti + 1),
                    labels_segments=pred
                )

        """ YOUR LINEAR SVM TRAINING CODE HERE """
        # Organizes samples as (n_sample, feature_vect)
        x = np.vstack(X_train)
        y = np.hstack(Y_train)
        svm.fit(x, y)

        """ LABEL THE TESTING SET AND PRINT RESULTS """
        scores_svm[fold] = svm.score(x, y)
        Y_pred = svm.predict(x)
        wrong_segments_svm.append(np.sum(y != Y_pred))
        # wrg_idx = np.flatnonzero(y != y_pred)

        fold += 1

    wrong_segments_crf = np.array(wrong_segments_crf)
    wrong_segments_svm = np.array(wrong_segments_svm)

    return (
        scores_svm,
        wrong_segments_svm,
        scores_crf,
        wrong_segments_crf,
    )


def show_global_results(scores_svm: np.ndarray,
                        wrong_segments_svm: np.ndarray,
                        scores_crf: np.ndarray,
                        wrong_segments_crf: np.ndarray,
                        total_segments: int,
                        ) -> None:
    """ Show global results """
    crf_score = 1.0 - wrong_segments_crf.mean() / float(total_segments)
    svm_score = 1.0 - wrong_segments_svm.mean() / float(total_segments)

    print('\nResults per fold ')
    print(f'Scores CRF : {scores_crf}')
    print(f'Scores SVM : {scores_svm}')
    print(f'Wrongs CRF : {wrong_segments_crf}')
    print(f'Wrongs SVM : {wrong_segments_svm}')
    print(' ')

    print(f'Final score CRF: {crf_score:.4}, {wrong_segments_crf.mean()} '
          f'wrong labels in total out of {total_segments}')

    print(f'Final score SVM: {svm_score:.4}, {wrong_segments_svm.mean()} '
          f'wrong labels in total out of {total_segments}')


def show_coefficients(weights: np.ndarray, num_features, num_labels):
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

    # Unary coefficients are the first [n_features * n_segment_types] coefs
    # Pairwise coefficients are the second [n_segment_types * n_segment_types] coefs
    unary_coef: np.ndarray = weights[0:(num_features * num_labels)]
    assert unary_coef.shape == (num_features * num_labels,)

    pairwise_coef: np.ndarray = weights[(num_features * num_labels):]
    assert pairwise_coef.shape == (num_labels * num_labels,)

    """ SHOW IMAGE OF THE LEARNED UNARY COEFFICIENTS, size (num_labels, num_features)"""
    """ use matshow() and colorbar()"""
    unary_coef = unary_coef.reshape(num_features, num_labels)

    plt.matshow(unary_coef)
    plt.colorbar()
    plt.xlabel('segment types')
    plt.ylabel('segment features')
    plt.title("Unary coefficients: importance of segment features")
    plt.show()

    """ SHOW IMAGE OF PAIRWISE COEFFICIENTS size (num_labels, num_labels)"""
    pairwise_coef = pairwise_coef.reshape(num_labels, num_labels)

    plt.matshow(pairwise_coef)
    plt.colorbar()
    plt.xlabel('segment types (current)')
    plt.ylabel('segment types (next)')
    plt.title("Pairwise coefficients: next segment type expectations")
    plt.show()


if __name__ == '__main__':
    add_gaussian_noise_to_features = False
    num_segments_per_jacket = 40
    num_features = 7
    sigma_noise = 0.1

    sheet = load_sheet()
    segments, labels_segments = load_segments(
        sheet,
        num_segments_per_jacket,
    )

    show_groundtruth(
        n=2,
        sheet=sheet,
        segments=segments,
        labels_segments=labels_segments
    )

    X, Y = prepare_data(
        segments,
        labels_segments,
        num_segments_per_jacket,
        num_features,
    )

    if add_gaussian_noise_to_features:
        X = add_gaussian_noise(X, sigma_noise)

    """
    DEFINE HERE YOUR GRAPHICAL MODEL AND CHOOSE ONE LEARNING METHOD
    (OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM)
    """
    svm = LinearSVC(dual=False, C=.1)

    model = ChainCRF()
    ssvm = FrankWolfeSSVM(model, C=0.1, max_iter=10)

    # With 5 in each fold we have 4 jackets for testing, 19 for training,
    # with 23 we have leave one out : 22 for training, 1 for testing
    results = compare_svm_and_ssvm(
        X=X, Y=Y, svm=svm, ssvm=ssvm, n_folds=5, segments=segments,
        show_labeling=False,
    )

    num_jackets = labels_segments.shape[0]
    num_labels = np.unique(np.ravel(labels_segments)).size
    total_segments = num_jackets * num_segments_per_jacket

    show_global_results(
        *results,
        total_segments=total_segments,
    )

    show_coefficients(
        weights=ssvm.w,
        num_features=num_features,
        num_labels=num_labels,
    )
