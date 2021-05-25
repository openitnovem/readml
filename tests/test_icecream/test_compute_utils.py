import numpy as np
import pandas as pd

from fbd_interpreter.icecream.compute_utils import (
    aggregate_series,
    compute_ale_agg_results,
    compute_ice_model_predictions,
    compute_ice_model_results_2D,
    compute_model_ale_results_2D,
    generate_fake_data,
    guess_model_predict_function,
    pivot_dataframe,
    sample_kmeans,
    sample_quantiles,
)
from fbd_interpreter.icecream.discretizer import FeatureDiscretizer

DUMMY_PREDICT = np.array([2, 3, 4])
DUMMY_PREDICT_PROBA = np.array([[1, 0.5, 0], [0, 0.5, 1]])


class DummyModel(object):
    """
    Dummy class that acts like a scikit-learn supervised learning model
    """

    def __init__(self, classifier: bool = True) -> None:
        self.predict = lambda x: DUMMY_PREDICT
        if classifier:
            self.classes_ = [0, 1]
            self.predict_proba = lambda x: DUMMY_PREDICT_PROBA


def test_generate_fake_data() -> None:
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    pd.testing.assert_frame_equal(
        generate_fake_data(data, "a", 1),
        pd.DataFrame([[1, 2], [1, 4]], columns=["a", "b"]),
    )


def test_aggregate_series() -> None:
    targets = pd.Series([0, 0, 1, 1])
    data = pd.Series([2, 2, 4, 4])
    fd = FeatureDiscretizer(data, 2)
    pd.testing.assert_series_equal(
        aggregate_series(fd, targets, "mean"),
        pd.Series(
            [0, 1],
            index=pd.Categorical(fd.categorical_feature.categories, ordered=True),
        ),
    )


def test_guess_model_predict_function() -> None:
    np.testing.assert_array_equal(
        guess_model_predict_function(DummyModel(False), use_classif_proba=False)(None),
        DUMMY_PREDICT,
    )
    np.testing.assert_array_equal(
        guess_model_predict_function(DummyModel(True), use_classif_proba=False)(None),
        DUMMY_PREDICT,
    )
    np.testing.assert_array_equal(
        guess_model_predict_function(DummyModel(True), use_classif_proba=True)(None),
        DUMMY_PREDICT_PROBA[:, 1],
    )


def test_compute_ale_agg_results() -> None:
    data = pd.DataFrame([2, 2, 4, 4], columns=["complex_data"])
    fd = FeatureDiscretizer(data["complex_data"], 2)
    function = DummyModel(True).predict
    pd.testing.assert_frame_equal(
        compute_ale_agg_results(data, fd, function),
        pd.DataFrame([[0.0, 0.0]], columns=fd.categorical_feature.categories),
    )


def test_compute_ice_model_predictions() -> None:
    data = pd.DataFrame([2, 2, 4, 4], columns=["complex_data"])
    fd = FeatureDiscretizer(data["complex_data"], 2)
    function = DummyModel(True).predict
    pd.testing.assert_frame_equal(
        compute_ice_model_predictions(data, fd, function),
        pd.DataFrame(
            [[2, 2], [3, 3], [4, 4]], columns=fd.categorical_feature.categories
        ),
    )


def test_pivot_dataframe() -> None:
    data = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=["a", "b", "t"])
    fd_a = FeatureDiscretizer(data["a"], 2)
    fd_b = FeatureDiscretizer(data["b"], 2)
    pivot_dataframe(data, "t", fd_a, fd_b, "mean", fill_value=0)
    pd.testing.assert_frame_equal(
        pivot_dataframe(data, "t", fd_a, fd_b, "mean", fill_value=0),
        pd.DataFrame(
            [[0, 0], [0, 1]],
            index=pd.CategoricalIndex(
                fd_b.categorical_feature.categories, ordered=True
            ),
            columns=pd.CategoricalIndex(
                fd_a.categorical_feature.categories, ordered=True
            ),
        ),
    )


def test_compute_results_ale_2D() -> None:
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    fd_a = FeatureDiscretizer(data["a"], 2)
    fd_b = FeatureDiscretizer(data["b"], 2)
    function = DummyModel(True).predict
    pd.testing.assert_frame_equal(
        compute_model_ale_results_2D(data, fd_a, fd_b, function),
        pd.DataFrame(
            [[0.0, 0.0], [0.0, 0.0]],
            index=fd_b.categorical_feature.categories,
            columns=fd_a.categorical_feature.categories,
        ),
    )


def test_compute_ice_model_results_2D() -> None:
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    fd_a = FeatureDiscretizer(data["a"], 2)
    fd_b = FeatureDiscretizer(data["b"], 2)
    function = DummyModel(True).predict
    pd.testing.assert_frame_equal(
        compute_ice_model_results_2D(data, fd_a, fd_b, function, "mean"),
        pd.DataFrame(
            [[3.0, 3.0], [3.0, 3.0]],
            index=fd_b.categorical_feature.categories,
            columns=fd_a.categorical_feature.categories,
        ),
    )


def test_sample_quantiles() -> None:
    data = pd.DataFrame([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    nb_rows = 3
    # not testing last outputs because they are cosmetic objects
    samples, counts, __, __ = sample_quantiles(data, nb_rows)
    pd.testing.assert_frame_equal(samples, pd.DataFrame([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(counts, np.array([11 / 3, 22 / 3, 11 / 3]))
    # function must also work with constant input
    data = pd.DataFrame([0.0, 0.0, 0.0, 0.0])
    nb_rows = 3
    samples, counts, __, __ = sample_quantiles(data, nb_rows)
    pd.testing.assert_frame_equal(samples, pd.DataFrame([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(counts, np.array([4 / 3, 8 / 3, 4 / 3]))


def test_sample_kmeans() -> None:
    data = pd.DataFrame([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    nb_rows = 3
    # not testing last outputs because they are cosmetic objects
    samples, counts, __, __ = sample_kmeans(data, nb_rows)
    np.testing.assert_array_equal(
        np.array(np.around(samples, 2)), np.array([[0.55], [0.15], [0.9]])
    )
    np.testing.assert_array_equal(counts, np.array([4, 4, 3]))
