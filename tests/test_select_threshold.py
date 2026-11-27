import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train import select_threshold  # noqa: E402


class FixedModel:
    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities, dtype=float)

    def predict_proba(self, _X):
        return np.column_stack([1 - self.probabilities, self.probabilities])


def test_picks_the_threshold_that_separates_the_classes():
    model = FixedModel([0.9, 0.8, 0.35, 0.3])
    threshold, f1 = select_threshold(model, None, np.array([1, 1, 0, 0]), thresholds=[0.2, 0.5, 0.85])
    assert threshold == 0.5
    assert f1 == 1.0


def test_falls_back_to_half_when_nothing_scores():
    model = FixedModel([0.1, 0.2])
    threshold, f1 = select_threshold(model, None, np.array([1, 1]), thresholds=[0.5, 0.9])
    assert (threshold, f1) == (0.5, 0)
