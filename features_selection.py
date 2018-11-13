from segment import Segment
import numpy as np


def all_features(s: Segment) -> np.array:
    # Note: some vars would need normalization on coefficient visualization
    return np.array((
        s.x0, s.y0, s.x1, s.y1,
        s.x0norm, s.y0norm, s.x1norm, s.y1norm,
        (s.x0norm + s.x1norm) / 2.,
        (s.y0norm + s.y1norm) / 2.,
        np.sqrt(
            (s.x0norm - s.x1norm) ** 2 + (s.y0norm - s.y1norm) ** 2),
        s.angle,
    ))


def default_features(s: Segment) -> np.array:
    return np.array((
        s.x0norm, s.y0norm, s.x1norm, s.y1norm,
        (s.x0norm + s.x1norm) / 2.,
        (s.y0norm + s.y1norm) / 2.,
        s.angle / (2 * np.pi),
    ))


def get_all_features():
    return (
        all_features,
        12,
        [
            'x0',
            'y0',
            'x1',
            'y1',
            'x0norm',
            'y0norm',
            'x1norm',
            'y1norm',
            'xlength',
            'ylength',
            'length',
            'angle',
        ])


def get_default_features():
    return (
        default_features,
        7,
        [
            'x0norm',
            'y0norm',
            'x1norm',
            'y1norm',
            'xlength',
            'ylength',
            'angle',
        ]
    )


features_options = {
    'all': get_all_features(),
    'default': get_default_features(),
}
