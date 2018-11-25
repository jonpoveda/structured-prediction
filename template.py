# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:15:45 2015

@author: joans
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import ExcelFile
from pathlib import Path
from pystruct.learners import OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM
from pystruct.models import ChainCRF, MultiClassClf
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from typing import List, Sequence, Tuple, Union, Callable, Optional, Iterable
from functools import partial

from feature_relevance import feature_relevance
from features_selection import features_options
from plot_segments import plot_segments


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
    wrongly_predicted_svm = None
    wrongly_predicted_crf = None

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
        wrongly_predicted_crf = np.flatnonzero(y != Y_pred)
        wrong_segments_crf.append(np.sum(Y_test != Y_pred))

        """ figure showing the result of classification of segments for
        each jacket in the testing part of present fold """
        if show_labeling:
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
        wrongly_predicted_svm = np.flatnonzero(y != Y_pred)
        wrong_segments_svm.append(np.sum(y != Y_pred))

        fold += 1

    wrong_segments_crf = np.array(wrong_segments_crf)
    wrong_segments_svm = np.array(wrong_segments_svm)

    return (
        scores_svm,
        wrong_segments_svm,
        scores_crf,
        wrong_segments_crf,
        wrongly_predicted_svm,
        wrongly_predicted_crf,
    )


def get_global_scores(*results, total_segments):
    scores_svm, wrong_segments_svm, scores_crf, wrong_segments_crf = results
    crf_score = 1.0 - wrong_segments_crf.mean() / float(total_segments)
    svm_score = 1.0 - wrong_segments_svm.mean() / float(total_segments)
    return svm_score, crf_score


def load_all_samples(num_segments_per_jacket):
    sheet = load_sheet(Path('man_jacket_hand_measures.xls'))
    samples, labels = load_segments(
        sheet=sheet,
        segments_dir=Path('segments'),
        num_segments_per_jacket=num_segments_per_jacket,
    )
    sheet_extra = load_sheet(
        Path('more_samples', 'man_jacket_hand_measures.xls'))
    samples_extra, labels_segments_extra = load_segments(
        sheet=sheet_extra,
        segments_dir=Path('more_samples/segments'),
        num_segments_per_jacket=num_segments_per_jacket,
    )
    samples = samples + samples_extra
    labels = np.concatenate((labels, labels_segments_extra))

    return samples, labels, sheet


def load_n_samples(num_segments_per_jacket: int, n: int):
    segments, labels_segments, sheet = load_all_samples(
        num_segments_per_jacket)
    return segments[0:n], labels_segments[0:n], sheet


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


def load_sheet(path: Union[str, Path]) -> pandas.DataFrame:
    """ Load the segments and the groundtruth for all jackets """
    with ExcelFile(path) as xl:
        # be careful, parse() just reads literals, does not execute formulas
        sheet = xl.parse(xl.sheet_names[0])
    return sheet


def plot_coefficients(weights: np.ndarray,
                      feature_names,
                      label_names,
                      experiment_name: Optional[str] = None,
                      ) -> None:
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

    # Normalize coefficients so can be treated as probabilities
    unary_coef = unary_coef - unary_coef.min(axis=1).reshape(-1, 1)
    unary_coef = unary_coef / unary_coef.sum(axis=1).reshape(-1, 1)

    plt.matshow(unary_coef, cmap='coolwarm')
    plt.title("Unary coefficients: importance of segment features"
              "\n[normalized as probabilities]", pad=32)
    plt.xlabel('segment types')
    plt.ylabel('segment features')
    plt.xticks(range(num_labels), label_names, rotation=15)
    plt.yticks(range(num_features), feature_names)
    plt.colorbar()

    if experiment_name is not None:
        fig_path = Path('logs', f'{experiment_name}_unary.png')
        plt.savefig(fig_path)

    plt.show()

    """ SHOW IMAGE OF PAIRWISE COEFFICIENTS size (num_labels, num_labels)"""
    pairwise_coef = pairwise_coef.reshape(num_labels, num_labels)

    pairwise_coef = pairwise_coef - pairwise_coef.min(axis=1).reshape(-1, 1)
    pairwise_coef = pairwise_coef / pairwise_coef.sum(axis=1).reshape(-1, 1)

    plt.matshow(pairwise_coef, cmap='coolwarm')
    plt.title("Pairwise coefficients: next segment type expectations"
              "\n[normalized as probabilities]", pad=32)
    plt.ylabel('segment types (next)')
    plt.xlabel('segment types (current)')
    plt.xticks(range(num_labels), label_names, rotation=15)
    plt.yticks(range(num_labels), label_names)
    plt.colorbar()

    if experiment_name is not None:
        fig_path = Path('logs', f'{experiment_name}_pairwise.png')
        plt.savefig(fig_path)

    plt.show()


def plot_groundtruth(sample_number: int,
                     sheet: pandas.DataFrame,
                     segments: Sequence,
                     labels_segments: np.ndarray
                     ) -> None:
    """ Show groundtruth for the n-jacket """
    fig = plot_segments(segments[sample_number], sheet.ide[sample_number],
                  labels_segments[sample_number])
    fig.savefig(Path('logs', f'groundtruth_sample_{sample_number}.png'))
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


def print_global_results(scores_svm: np.ndarray,
                         wrong_segments_svm: np.ndarray,
                         scores_crf: np.ndarray,
                         wrong_segments_crf: np.ndarray,
                         svm_score: float,
                         crf_score: float,
                         total_segments: int,
                         ) -> None:
    """ Show global results """

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


def run(add_gaussian_noise_to_features=False,
        num_segments_per_jacket=40,
        feature_set='default',
        sigma_noise=0.1,
        sample_loader: Callable = None,
        learning_method: Callable = None,
        features: Iterable[int] = None,
        show_groundtruth=False,
        show_global_results=False,
        show_coefficients=False,
        experiment_name: Optional[str] = None,
        ) -> Tuple[float, float]:
    if sample_loader is None:
        sample_loader = load_all_samples

    segments, labels_segments, sheet = sample_loader(num_segments_per_jacket)

    if show_groundtruth:
        plot_groundtruth(
            sample_number=2,
            sheet=sheet,
            segments=segments,
            labels_segments=labels_segments
        )

    X, Y = prepare_data(
        segments,
        labels_segments,
        num_segments_per_jacket,
        feature_selection=feature_set,
    )

    def pick_features(arr: np.ndarray, features: Iterable[int]):
        return arr[:, :, features]

    if features is not None:
        X = pick_features(X, features)

    num_features = X.shape[2]

    if add_gaussian_noise_to_features:
        X = add_gaussian_noise(X, sigma_noise)

    """
    DEFINE HERE YOUR GRAPHICAL MODEL AND CHOOSE ONE LEARNING METHOD
    (OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM)
    """
    svm = LinearSVC(dual=False, C=.1)

    model = ChainCRF()
    ssvm = learning_method(model, C=0.1, max_iter=10)

    # With 5 in each fold we have 4 jackets for testing, 19 for training,
    # with 23 we have leave one out : 22 for training, 1 for testing
    results = compare_svm_and_ssvm(
        X=X, Y=Y, svm=svm, ssvm=ssvm, n_folds=5, segments=segments,
        show_labeling=False,
    )

    num_jackets = labels_segments.shape[0]
    num_labels = np.unique(np.ravel(labels_segments)).size
    total_segments = num_jackets * num_segments_per_jacket

    svm_score, crf_score = get_global_scores(
        *results, total_segments=total_segments
    )

    if experiment_name is not None:
        data_path = Path('logs', f'{experiment_name}_scores.json')
        with data_path.open('w') as file:
            json.dump(
                {
                    'svm_score': svm_score,
                    'crf_score': crf_score,
                },
                file,
                indent=2,
            )

    if show_global_results:
        print_global_results(
            *results,
            svm_score=svm_score,
            crf_score=crf_score,
            total_segments=total_segments,
        )

    if show_coefficients:
        _, _, features_names = features_options[feature_set]

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

        plot_coefficients(
            weights=ssvm.w,
            feature_names=features_names,
            label_names=label_names,
            experiment_name=experiment_name,
        )

    return svm_score, crf_score


def run_test_noise(num_segments_per_jacket, features, experiment_name: str):
    svm_scores = []
    crf_scores = []
    sigmas = np.arange(0, 3, 0.1).tolist()

    for sigma_noise in sigmas:
        print(f'Running with sigma = {sigma_noise}')
        scores = run(
            num_segments_per_jacket=num_segments_per_jacket,
            feature_set=features,
            add_gaussian_noise_to_features=True,
            sigma_noise=sigma_noise,
            sample_loader=load_all_samples,
            learning_method=FrankWolfeSSVM,
            experiment_name=experiment_name,
        )
        svm_score, crf_score = scores

        svm_scores.append(svm_score)
        crf_scores.append(crf_score)

    fig_path = Path('logs', f'{experiment_name}_noise.png')
    plt.plot(sigmas, svm_scores, label='LinearSVC')
    plt.plot(sigmas, crf_scores, label='FrankWolfeSSVM')
    plt.xlabel('Noise (sigma)')
    plt.ylabel('Accuracy')
    plt.autoscale(axis='x', tight=True)
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Effect of noise')
    plt.savefig(fig_path)
    plt.show()

    data_path = Path('logs', f'{experiment_name}_noise.json')
    with data_path.open('w') as file:
        json.dump(
            {
                'sigmas': sigmas,
                'svm_scores': svm_scores,
                'crf_scores': crf_scores,
            },
            file,
            indent=2,
        )

    return sigmas, svm_scores, crf_scores


def run_test_number_of_samples(num_segments_per_jacket,
                               features,
                               experiment_name: str):
    svm_scores = []
    crf_scores = []
    sigmas = np.arange(0, 3, 0.1).tolist()

    loaders = []
    number_of_samples = list(range(105, 5, -10))
    for num in number_of_samples:
        loaders.append(
            partial(load_n_samples, n=num)
        )

    for sample_loader in loaders:
        print(f'Running with loader = {sample_loader}')
        scores = run(
            num_segments_per_jacket=num_segments_per_jacket,
            feature_set=features,
            sample_loader=sample_loader,
            learning_method=FrankWolfeSSVM,
            experiment_name=experiment_name,
        )
        svm_score, crf_score = scores

        svm_scores.append(svm_score)
        crf_scores.append(crf_score)

    fig_path = Path('logs', f'{experiment_name}_samples.png')
    plt.plot(number_of_samples, svm_scores, label='LinearSVC')
    plt.plot(number_of_samples, crf_scores, label='FrankWolfeSSVM')
    plt.xlabel('Number of samples used')
    plt.ylabel('Accuracy')
    plt.autoscale(axis='x', tight=True)
    plt.ylim(0.4, 1)
    plt.legend()
    plt.title('Effect of number of samples')
    plt.savefig(fig_path)
    plt.show()

    data_path = Path('logs', f'{experiment_name}_samples.json')
    with data_path.open('w') as file:
        json.dump(
            {
                'number_of_samples': number_of_samples,
                'svm_scores': svm_scores,
                'crf_scores': crf_scores,
            },
            file,
            indent=2,
        )

    return sigmas, svm_scores, crf_scores


def run_test_learning_method(num_segments_per_jacket,
                             features,
                             experiment_name,
                             ) -> Tuple[List[str], float, float, float]:
    """ Compares different learning methods

    SVM is run always so its result is averaged.
    """
    methods = [
        OneSlackSSVM, NSlackSSVM, FrankWolfeSSVM,
    ]
    method_names = [str(method.__name__) for method in methods]

    svm_scores = []
    crf_scores = []

    # partial(load_n_segments, n=num)

    for method in methods:
        print(f'Running with method = {method.__name__}')
        scores = run(
            num_segments_per_jacket=num_segments_per_jacket,
            feature_set=features,
            sample_loader=load_all_samples,
            learning_method=method,
            experiment_name=experiment_name,
        )
        svm_score, crf_score = scores

        svm_scores.append(svm_score)
        crf_scores.append(crf_score)

    one_slack_score, n_slack_score, frankwolfe_score = crf_scores
    svm_score = np.mean(svm_scores)

    fig_path = Path('logs', f'{experiment_name}_method.png')
    plt.bar('LinearSVC', svm_score, label='LinearSVC')
    plt.bar('OneSlackSSVM', one_slack_score, label='OneSlackSSVM')
    plt.bar('NSlackSSVM', n_slack_score, label='NSlackSSVM')
    plt.bar('FrankWolfeSSVM', frankwolfe_score, label='FrankWolfeSSVM')
    plt.xlabel('Learning method')
    plt.ylabel('Accuracy')
    plt.autoscale(axis='x', tight=True)
    plt.ylim(0.80, 1)
    plt.title('Comparing learning methods')
    plt.savefig(fig_path)
    plt.show()

    data_path = Path('logs', f'{experiment_name}_method.json')
    with data_path.open('w') as file:
        json.dump(
            {
                'method_names': method_names,
                'one_slack_score': one_slack_score,
                'n_slack_score': n_slack_score,
                'frankwolfe_score': frankwolfe_score,
            },
            file,
            indent=2,
        )
    return method_names, one_slack_score, n_slack_score, frankwolfe_score




def run_test_features(num_segments_per_jacket,
                      features,
                      experiment_name):
    unary_coeff_default_experiment = np.array([
        [0.00685162, 0.13670038, 0., 0.13396857, 0.00342581, 0.13533447,
         0.08383054, 0.09523605, 0.09793283, 0.08699478, 0.12860954,
         0.09111541],
        [0.12336596, 0.09957966, 0.13677306, 0., 0.06122748, 0.06338093,
         0.09900027, 0.03169047, 0.01410955, 0.14888256, 0.07425956,
         0.14773051],
        [0.0084382, 0.13566063, 0.02305881, 0., 0.05875059, 0.02639469,
         0.16681362, 0.00441264, 0.1127821, 0.01540366, 0.28474956,
         0.1635355, ],
        [0.03509838, 0.06866554, 0.06122099, 0.09813618, 0.04815969,
         0.04143554, 0.15958555, 0.11057723, 0.16135967, 0., 0.16047261,
         0.05528862],
        [0.08327444, 0.04525033, 0.05142011, 0.12703061, 0.08901612,
         0.08614047, 0.07021811, 0.12041581, 0.09406804, 0.1328481, 0.,
         0.10031785],
        [0.07472008, 0.10568236, 0., 0.09094891, 0.10052445, 0.08653539,
         0.08219606, 0.08874215, 0.09136026, 0.12239117, 0.04160312,
         0.11529605],
        [0.05937339, 0.18471646, 0.01332578, 0.16094225, 0.14690756, 0.,
         0.0978974, 0.01056175, 0.12406503, 0.00528087, 0.11098122, 0.08594829
         ]]
    )
    feature_sorted_by_relevance, _ = feature_relevance(
        unary_coeff_default_experiment)

    feature_lists = []
    for idx, feature in enumerate(feature_sorted_by_relevance, start=1):
        feature_lists.append(feature_sorted_by_relevance[0:idx])

    number_of_features = [len(_) for _ in feature_lists]
    print(feature_lists)
    print(number_of_features)
    # fold == [(2,), (2, 6), (2, 6, 3), (2, 6, 3, 0), (2, 6, 3, 0, 1),
    #           (2, 6, 3, 0, 1, 4), (2, 6, 3, 0, 1, 4, 5)]

    # TODO (jonatan@adsmurai.com) do a 'run' selecting the features in the 'fold'
    svm_scores = []
    crf_scores = []

    for feature_list in feature_lists:
        print(f'Running with features = {feature_list}')
        scores = run(
            num_segments_per_jacket=num_segments_per_jacket,
            feature_set=features,
            sample_loader=load_all_samples,
            learning_method=FrankWolfeSSVM,
            experiment_name=experiment_name,
            features=feature_list,
        )
        svm_score, crf_score = scores

        svm_scores.append(svm_score)
        crf_scores.append(crf_score)

    fig_path = Path('logs', f'{experiment_name}_features.png')
    plt.plot(number_of_features, svm_scores, label='LinearSVC')
    plt.plot(number_of_features, crf_scores, label='FrankWolfeSSVM')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.xlim(7, 1)
    plt.ylim(0.40, 1)
    plt.title('Effect of reducing the number of features')
    plt.legend()
    plt.savefig(fig_path)
    plt.show()

    data_path = Path('logs', f'{experiment_name}_features.json')
    with data_path.open('w') as file:
        json.dump(
            {
                'features_used': feature_lists,
                'svm_scores': svm_scores,
                'crf_scores': crf_scores,
            },
            file,
            indent=2,
        )
    return svm_scores, crf_scores


if __name__ == '__main__':
    Path('logs').mkdir(exist_ok=True)

    num_segments_per_jacket = 40
    features = 'default'

    run(
        add_gaussian_noise_to_features=False,
        num_segments_per_jacket=num_segments_per_jacket,
        feature_set='default',
        sample_loader=load_all_samples,
        learning_method=FrankWolfeSSVM,
        show_groundtruth=True,
        show_global_results=True,
        show_coefficients=True,
        experiment_name='FrankWolfeSSVM',
    )

    run_test_noise(num_segments_per_jacket,
                   features,
                   experiment_name='noise_FrankWolfeSSVM')

    run_test_number_of_samples(num_segments_per_jacket,
                               features,
                               experiment_name='segments_FrankWolfeSSVM')

    run_test_learning_method(num_segments_per_jacket,
                             features,
                             experiment_name='learning_methods')

    run_test_features(num_segments_per_jacket,
                      features,
                      experiment_name='features_FrankWolfeSSVM')
