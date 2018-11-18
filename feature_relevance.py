import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def plain_histogram_cost(histogram: np.ndarray):
    plain = np.full_like(histogram, fill_value=1 / histogram.shape[0])
    diff = histogram - plain
    energy = diff ** 2
    return energy.sum()


def feature_relevance(coeffs: np.array, sort: bool = True):
    """ Returns features sorted by decreasing relevance and their relevance

    Args:
        coeffs: an matrix of unary coefficients (num_feats, num_segments)
    """
    feature_costs = [plain_histogram_cost(feature) for feature in coeffs]
    pairs = [(cost, idx) for idx, cost in enumerate(feature_costs)]
    if sort:
        pairs.sort(reverse=True)
    relevances, feature_by_relevance = zip(*pairs)

    return feature_by_relevance, relevances


if __name__ == '__main__':
    plot_histograms = False

    unary_coeff = np.array([
        [0.00685162, 0.13670038, 0., 0.13396857, 0.00342581, 0.13533447,
         0.08383054, 0.09523605, 0.09793283, 0.08699478, 0.12860954,
         0.09111541],
        [0.12336596, 0.09957966, 0.13677306, 0., 0.06122748, 0.06338093,
         0.09900027, 0.03169047, 0.01410955, 0.14888256, 0.07425956,
         0.14773051],
        [0.0084382, 0.13566063, 0.02305881, 0., 0.05875059, 0.02639469,
         0.16681362,
         0.00441264, 0.1127821, 0.01540366, 0.28474956, 0.1635355, ],
        [0.03509838, 0.06866554, 0.06122099, 0.09813618, 0.04815969,
         0.04143554,
         0.15958555, 0.11057723, 0.16135967, 0., 0.16047261, 0.05528862],
        [0.08327444, 0.04525033, 0.05142011, 0.12703061, 0.08901612,
         0.08614047,
         0.07021811, 0.12041581, 0.09406804, 0.1328481, 0., 0.10031785],
        [0.07472008, 0.10568236, 0., 0.09094891, 0.10052445, 0.08653539,
         0.08219606, 0.08874215, 0.09136026, 0.12239117, 0.04160312,
         0.11529605],
        [0.05937339, 0.18471646, 0.01332578, 0.16094225, 0.14690756, 0.,
         0.0978974,
         0.01056175, 0.12406503, 0.00528087, 0.11098122, 0.08594829
         ]]
    )

    feature_by_relevance, relevances = feature_relevance(unary_coeff)

    print(f'Features sorted by decreasing relevance: {feature_by_relevance}\n'
          f'Relevances: {relevances}')

    # Plot feature relevances
    feature_by_relevance, relevances = feature_relevance(unary_coeff, sort=False)
    plt.figure()
    plt.fill_between(feature_by_relevance, relevances)
    plt.xlabel('Feature number')
    plt.ylabel('Energy')
    plt.autoscale(axis='x', tight=True)
    plt.ylim(0)
    plt.title('Feature relevances')
    plt.savefig(Path('logs', 'feature_relevances.png'))
    plt.show()

    if plot_histograms:
        for i, feature in enumerate(unary_coeff):
            plt.bar(range(1, len(feature) + 1), feature,
                    color='b',
                    align='center')
            plt.title(f'Feature {i}')
            plt.xlabel('Segments')
            plt.autoscale(axis='both')

        for i, segment in enumerate(unary_coeff.T):
            plt.bar(range(1, len(segment) + 1), segment,
                    color='r')
            plt.title(f'Segment {i}')
            plt.xlabel('Features')
            plt.autoscale(axis='both')
            plt.show()
