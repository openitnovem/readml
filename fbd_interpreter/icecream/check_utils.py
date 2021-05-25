"""
FBDTools library - icecream package
This module contains useful functions for checking user input and
data consistency
"""

import warnings
from typing import Any, Dict, List, Optional, Sized, Union

import numpy as np
import pandas as pd


def check_clip_quantile(clip_quantile: float) -> None:
    if clip_quantile >= 0.5:
        raise ValueError("Quantile for clipping must be strictly below 0.5")


def check_model_predict(model: Any, use_classif_proba: bool) -> None:
    if model is not None:
        if not hasattr(model, "predict"):
            raise AttributeError("Model does not have a predict method")
        if hasattr(model, "classes_") and use_classif_proba:
            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    "Model does not have a predict_proba method, please use predict"
                )
            if len(model.classes_) > 2:
                raise NotImplementedError(
                    "Multiclass predict_proba not supported, please use predict"
                )


def check_data_features(data: pd.DataFrame, feature_names: List[str]) -> None:
    for name in feature_names:
        if name not in data:
            raise ValueError("Feature '{}' not present in dataset".format(name))


def check_bins(bins: Union[Dict[str, Union[int, Sized, None]], int]) -> None:
    if not isinstance(bins, (dict, int, np.integer)):
        raise ValueError(
            "Incorrect bins definition, please define bins"
            " as a dictionary or a single integer"
        )


def check_input_in_list(value: str, possible: List[str], arg_name: str = "") -> None:
    if value not in possible:
        raise ValueError(
            "Incorrect value '{}' for argument {}, possible values are {}".format(
                value, arg_name, ", ".join(possible)
            )
        )


def check_sized_lengths(
    ref: Union[pd.Series, pd.DataFrame], *args: Union[pd.Series, pd.DataFrame]
) -> None:
    for arg in args:
        if arg is not None and len(arg) != len(ref):
            raise ValueError("Pandas/Numpy objects must have the same length")


def check_ale(data: pd.DataFrame, features_names: List[str], use_ale: bool) -> None:
    if use_ale:
        for name in features_names:
            if data[name].dtypes == object:
                raise NotImplementedError(
                    "ALE for categorical features not supported yet"
                )


def conduct_full_check(
    data: pd.DataFrame,
    feature_names: List[str],
    bins: Union[Dict[str, Union[int, Sized, None]], int],
    model: Optional[Any],
    predictions: Optional[Sized],
    targets: Optional[Sized],
    aggfunc: str,
    available_aggfuncs: List[str],
    use_classif_proba: bool,
    clip_quantile: float,
    use_ale: bool = False,
) -> None:
    """Conducts full check of partial dependencies classes inputs."""
    check_data_features(data, feature_names)
    check_bins(bins)
    check_clip_quantile(clip_quantile)
    check_model_predict(model, use_classif_proba)
    check_sized_lengths(data, predictions, targets)
    check_ale(data, feature_names, use_ale)
    if aggfunc not in available_aggfuncs:
        warnings.warn(
            "Aggregation function '{}' not officially supported"
            ", please use one of the following: {}".format(
                aggfunc, ", ".join(available_aggfuncs)
            )
        )
