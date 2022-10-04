# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from readml.icecream import options
from readml.icecream.discretizer import FeatureDiscretizer


def test_categorical_feature() -> None:
    feature_name = "complex_categorical_data"
    series = pd.Series(["a", "c", "a", "b"], name=feature_name)
    categorical_series = pd.Categorical(
        ["a", "c", "a", "b"], categories=["a", "b", "c"]
    )
    feature = FeatureDiscretizer(series)
    assert (feature.categorical_feature == categorical_series).all()
    assert feature.widths is None
    assert (feature.centers == np.array(["a", "b", "c"])).all()
    assert (feature.counts == categorical_series.value_counts()).all()
    assert feature.name == feature_name
    assert len(feature) == 3


def test_continuous_feature() -> None:
    feature_name = "complex_continuous_data"
    bins = 2
    series = pd.Series([0, 0, 0, 1, 0, 1, 1, 1, 1], name=feature_name)
    categorical_series = pd.Categorical(pd.cut(series, bins=bins))
    feature = FeatureDiscretizer(series, bins=bins)
    assert (feature.categorical_feature == categorical_series).all()
    assert (feature.widths == np.array([0.501, 0.5])).all()
    assert (feature.centers == np.array([0.2495, 0.75])).all()
    assert (feature.counts == categorical_series.value_counts()).all()
    assert feature.name == feature_name
    assert len(feature) == 2


def test_guess_bins() -> None:
    assert FeatureDiscretizer._guess_bins(pd.Series()) == 0
    assert FeatureDiscretizer._guess_bins(pd.Series(["a", "b", "c"])) == 0
    assert FeatureDiscretizer._guess_bins(pd.Categorical([0, 0, 1])) == 0
    assert FeatureDiscretizer._guess_bins(pd.Series([0, 0, 1])) == 0
    assert (
        FeatureDiscretizer._guess_bins(pd.Series(np.random.randn(100)))
        == options.default_number_bins
    )


def test_clip_values() -> None:
    series = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    assert (FeatureDiscretizer._clip_values(series, 0.0) == series).all()
    assert (
        FeatureDiscretizer._clip_values(series, 0.1)
        == pd.Series([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9])
    ).all()
    with pytest.raises(Exception) as e_info:
        _ = FeatureDiscretizer._clip_values(series, 0.5)
    assert isinstance(e_info.value, ValueError)
    with pytest.raises(Exception) as e_info:
        _ = FeatureDiscretizer._clip_values(series, 1.0)
    assert isinstance(e_info.value, ValueError)
