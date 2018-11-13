# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
import pandas
import matplotlib.pyplot as plt
import numpy as np

from pandas import ExcelFile
from pathlib import Path

from pystruct.models import ChainCRF, MultiClassClf
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from typing import Tuple, Union, Sequence, List

from features_selection import features_options
from plot_segments import plot_segments
from segment import Segment


def load_sheet(path: Union[str, Path]) -> pandas.DataFrame:
    """ Load the segments and the groundtruth for all jackets """
    with ExcelFile(path) as xl:
        # be careful, parse() just reads literals, does not execute formulas
        sheet = xl.parse(xl.sheet_names[0])
    return sheet


def load_segments(sheet: pandas.DataFrame,
                  segments_dir: Path,
                  num_segments_per_jacket: int,
                  ) -> Tuple[List, np.ndarray]:
    """ Loads segments present in a sheet from a directory

    Returns:
        a collection of jackets (containing arrays of Segments) and its labels
    """
    it = sheet.iterrows()
    labels_segments = []
    segments = []

    for row in it:
        ide = row[1]['ide']
        segments.append(np.load(
            str(segments_dir.joinpath(ide + '_front.npy')),
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
                 feature_selection: str = 'default',
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
    get_features, num_features, _ = features_options[feature_selection]

    Y = labels_segments
    num_jackets = labels_segments.shape[0]
    X = np.zeros((num_jackets, num_segments_per_jacket, num_features))

    """ set the features """
    for i, jacket_segments in enumerate(segments):
        for j, s in enumerate(jacket_segments):
            X[i, j, 0:num_features] = get_features(s)

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


def show_coefficients(weights: np.ndarray, feature_names, label_names):
    num_labels = len(label_names)
    num_features = len(feature_names)

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
    plt.title("Unary coefficients: importance of segment features", pad=32)
    plt.xlabel('segment types')
    plt.ylabel('segment features')
    plt.xticks(range(num_labels), label_names, rotation=15)
    plt.yticks(range(num_features), feature_names)
    plt.colorbar()
    plt.show()

    """ SHOW IMAGE OF PAIRWISE COEFFICIENTS size (num_labels, num_labels)"""
    pairwise_coef = pairwise_coef.reshape(num_labels, num_labels)

    plt.matshow(pairwise_coef)
    plt.title("Pairwise coefficients: next segment type expectations", pad=32)
    plt.ylabel('segment types (next)')
    plt.xlabel('segment types (current)')
    plt.xticks(range(num_labels), label_names, rotation=15)
    plt.yticks(range(num_labels), label_names)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    add_gaussian_noise_to_features = False
    num_segments_per_jacket = 40
    features = 'default'
    sigma_noise = 0.1

    sheet = load_sheet(Path('man_jacket_hand_measures.xls'))
    segments, labels_segments = load_segments(
        sheet=sheet,
        segments_dir=Path('segments'),
        num_segments_per_jacket=num_segments_per_jacket,
    )

    sheet_extra = load_sheet(
        Path('more_samples', 'man_jacket_hand_measures.xls'))
    segments_extra, labels_segments_extra = load_segments(
        sheet=sheet_extra,
        segments_dir=Path('more_samples/segments'),
        num_segments_per_jacket=num_segments_per_jacket,
    )

    segments = segments + segments_extra
    labels_segments = np.concatenate((labels_segments, labels_segments_extra))

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
        feature_selection=features,
    )

    num_features = X.shape[2]

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

    _, _, features_names = features_options[features]

    label_names = [
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

    show_coefficients(
        weights=ssvm.w,
        feature_names=features_names,
        label_names=label_names,
    )
