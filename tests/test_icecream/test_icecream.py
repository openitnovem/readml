import pandas as pd

from fbd_interpreter.icecream.icecream import IceCream

DATA = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 1, 1, 1, 2, 2, 2, 2]})
FEATURES = ["a", "b"]
PREDICTIONS = [0, 0, 0, 0, 1, 1, 1, 1]
TARGETS = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])


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
        self.predict_proba = lambda x: None


def test_icecream_run() -> None:
    model = DummyModel()
    pdp = IceCream(
        data=DATA,
        feature_names=FEATURES,
        bins=2,
        model=model,
        targets=TARGETS,
        aggfunc="mean",
        use_classif_proba=False,
        clip_quantile=0.0,
    )

    for i, name in enumerate(FEATURES):
        assert (
            (
                pdp.predictions[name]
                == pd.DataFrame(
                    {
                        key: PREDICTIONS
                        for key in pdp.features[i].categorical_feature.categories
                    }
                )
            )
            .all()
            .all()
        )
        assert (
            pdp.agg_predictions[name]
            == pd.Series(
                [0.5, 0.5], index=pdp.features[i].categorical_feature.categories
            )
        ).all()
        assert (
            pdp.agg_targets[name]
            == pd.Series(
                [0.0, 1.0], index=pdp.features[i].categorical_feature.categories
            )
        ).all()
