import numpy as np


class SoftVotingEnsemble:
    """Soft voting ensemble that averages class probabilities across tree & boosting models."""
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        return self

    def predict(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return np.argmax(probs, axis=1)
