import os

import pandas as pd

from fbd_interpreter.explainers.ml.explain_ml import ExplainML
from fbd_interpreter.logger import ROOT_DIR

FEATURES = ["a", "b"]
PREDICTIONS = [0, 0, 0, 0, 1, 1, 1, 1]
TARGETS = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
DATA = pd.DataFrame(
    {"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 1, 1, 1, 2, 2, 2, 2], "target": TARGETS}
)


class DummyModel(object):
    """
    Dummy class that acts like a scikit-learn supervised learning model.
    Always makes the same predictions.
    """

    def __init__(
        self,
    ) -> None:
        self.predict = lambda x: PREDICTIONS
        self.classes_ = [0, 1]
        self.predict_proba = lambda x: [[0.9, 0.1]]


# TODO
"""
def test_global_pdp_ice() -> None:
    interpreter = ExplainML(
        model=DummyModel(),
        task_name="classification",
        tree_based_model=False,
        features_name=FEATURES,
        features_to_interpret=FEATURES,
        target_col="target",
        out_path=os.path.join(ROOT_DIR, "../outputs/tests"),
    )
    interpreter.global_pdp_ice(DATA)
"""
