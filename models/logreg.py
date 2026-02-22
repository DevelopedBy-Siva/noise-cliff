from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from config import LOGREG_CONFIG


class LogRegModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=LOGREG_CONFIG["max_features"],
                        ngram_range=LOGREG_CONFIG["ngram_range"],
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=LOGREG_CONFIG["max_iter"],
                        C=LOGREG_CONFIG["C"],
                        solver=LOGREG_CONFIG["solver"],
                    ),
                ),
            ]
        )

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

    def get_loss_per_sample(self, texts, labels):
        """
        Returns a per-sample loss score using log loss.
        Higher score means the model is more confused about that sample.
        Used by the cleaning strategies to rank samples by suspiciousness.
        """
        import numpy as np

        proba = self.predict_proba(texts)
        eps = 1e-9
        losses = []
        for i, label in enumerate(labels):
            p = np.clip(proba[i][label], eps, 1 - eps)
            losses.append(-np.log(p))
        return losses
