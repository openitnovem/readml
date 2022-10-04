# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

from readml.icecream.compute_utils import (
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
from readml.icecream.discretizer import FeatureDiscretizer

DUMMY_REG_PREDICT = np.array([2, 3, 4])
DUMMY_BINARY_PREDICT_PROBA = np.array([[1, 0], [0.5, 0.5], [0, 1]])
DUMMY_BINARY_PREDICT = np.array([0, 0, 1])
DUMMY_MULTICLASS_PREDICT_PROBA = np.array([[0.8, 0.2, 0], [0, 0.5, 0.5]])
DUMMY_MULTICLASS_PREDICT = np.array([0, 2])


class DummyModel(object):
    """
    Dummy class that acts like a scikit-learn supervised learning model
    """

    def __init__(self, type: str) -> None:
        if type == "reg":
            self.predict = lambda x: DUMMY_REG_PREDICT
        elif type == "binary":
            self.classes_ = np.array(range(DUMMY_BINARY_PREDICT_PROBA.shape[1]))
            self.predict = lambda x: DUMMY_BINARY_PREDICT
            self.predict_proba = lambda x: DUMMY_BINARY_PREDICT_PROBA
        elif type == "multiclass":
            self.classes_ = np.array(range(DUMMY_MULTICLASS_PREDICT_PROBA.shape[1]))
            self.predict = lambda x: DUMMY_MULTICLASS_PREDICT
            self.predict_proba = lambda x: DUMMY_MULTICLASS_PREDICT_PROBA


def test_generate_fake_data() -> None:
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    pd.util.testing.assert_frame_equal(
        generate_fake_data(data, "a", 1),
        pd.DataFrame([[1, 2], [1, 4]], columns=["a", "b"]),
    )


def test_aggregate_series() -> None:
    targets = pd.Series([0, 0, 1, 1])
    data = pd.Series([2, 2, 4, 4])
    fd = FeatureDiscretizer(data, 2)
    pd.util.testing.assert_series_equal(
        aggregate_series(fd, targets, "mean"),
        pd.Series(
            [0.0, 1.0],
            index=pd.Categorical(fd.categorical_feature.categories, ordered=True),
        ),
    )


def test_guess_model_predict_function() -> None:
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(DummyModel("reg"), use_classif_proba=False)(None),
        DUMMY_REG_PREDICT,
    )
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(DummyModel("binary"), use_classif_proba=False)(
            None
        ),
        DUMMY_BINARY_PREDICT,
    )
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(DummyModel("binary"), use_classif_proba=True)(
            None
        ),
        DUMMY_BINARY_PREDICT_PROBA[:, 1],
    )
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(
            DummyModel("multiclass"), use_classif_proba=False, class_name=1
        )(None),
        DUMMY_MULTICLASS_PREDICT,
    )
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(
            DummyModel("multiclass"), use_classif_proba=True, class_name=1
        )(None),
        DUMMY_MULTICLASS_PREDICT_PROBA[:, 1],
    )
    pd.util.testing.assert_numpy_array_equal(
        guess_model_predict_function(
            DummyModel("multiclass"), use_classif_proba=True, class_name=2
        )(None),
        DUMMY_MULTICLASS_PREDICT_PROBA[:, 2],
    )


def test_compute_ale_agg_results() -> None:
    data = pd.DataFrame([2, 2, 4, 4], columns=["complex_data"])
    fd = FeatureDiscretizer(data["complex_data"], 2)
    function = DummyModel("binary").predict
    pd.util.testing.assert_frame_equal(
        compute_ale_agg_results(data, fd, function, add_mean=False),
        pd.DataFrame([[0.0, 0.0]], columns=fd.categorical_feature.categories),
    )


def test_compute_ice_model_predictions() -> None:
    data = pd.DataFrame([2, 2, 4, 4], columns=["complex_data"])
    fd = FeatureDiscretizer(data["complex_data"], 2)
    function = DummyModel("reg").predict
    pd.util.testing.assert_frame_equal(
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
    pd.util.testing.assert_frame_equal(
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
    function = DummyModel("reg").predict
    pd.util.testing.assert_frame_equal(
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
    function = DummyModel("reg").predict
    pd.util.testing.assert_frame_equal(
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
    pd.util.testing.assert_frame_equal(samples, pd.DataFrame([0.0, 0.5, 1.0]))
    pd.util.testing.assert_numpy_array_equal(counts, np.array([11 / 3, 22 / 3, 11 / 3]))
    # function must also work with constant input
    data = pd.DataFrame([0.0, 0.0, 0.0, 0.0])
    nb_rows = 3
    samples, counts, __, __ = sample_quantiles(data, nb_rows)
    pd.util.testing.assert_frame_equal(samples, pd.DataFrame([0.0, 0.0, 0.0]))
    pd.util.testing.assert_numpy_array_equal(counts, np.array([4 / 3, 8 / 3, 4 / 3]))


def test_sample_kmeans() -> None:
    data = pd.DataFrame([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    nb_rows = 3
    # not testing last outputs because they are cosmetic objects
    samples, counts, __, __ = sample_kmeans(data, nb_rows)
    pd.util.testing.assert_numpy_array_equal(
        np.array(np.around(samples, 2)), np.array([[0.55], [0.15], [0.9]])
    )
    pd.util.testing.assert_numpy_array_equal(counts, np.array([4, 4, 3]))
