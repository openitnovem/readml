"""
FBDTools library - icecream package
This module contains useful functions to compute and aggregate predictions and
target values.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import options
from .discretizer import FeatureDiscretizer


def guess_model_predict_function(
    model: Any,
    use_classif_proba: bool,
    class_name: Optional[Union[int, str]] = None,
) -> Callable:
    """
    Returns model prediction method guessed from model properties.
    Multi-class classification not implemented yet.
    Parameters
    ----------
    model : scikit-learn model
        Model to compute predictions, `model.predict()` must work
    use_classif_proba : bool
        If True, use `predict_proba` for positive class as model output,
        only used if model is a classifier
    class_name: Optional[Union[int, str]]
        Name of the class of interest. It is likely to be an integer or a string,
        it depends on the type of the target vector type.

    Returns
    -------
    function : Callable
        Prediction method for direct use on data
    """
    if hasattr(model, "classes_"):
        if len(model.classes_) > 2:  # multiclass
            if not class_name:
                raise ValueError("class_name must be provided for multiclass model.")
            if use_classif_proba:
                if not class_name in model.classes_:
                    raise ValueError(
                        f"Provided class_name, {class_name}, "
                        f"should be a valid class among {model.classes_}."
                    )
                class_id = np.argwhere(model.classes_ == class_name)
                assert len(class_id) == 1  # only one class with this name
                class_id = class_id[0]
                return lambda x: model.predict_proba(x)[:, class_id[0]]
            return model.predict
        else:  # binary class
            if use_classif_proba:
                return lambda x: model.predict_proba(x)[:, 1]
            return model.predict
    # regression
    return model.predict


def compute_ale_agg_results(
    data: pd.DataFrame,
    feature: FeatureDiscretizer,
    predict_function: Callable,
    add_mean: bool = False,
) -> pd.DataFrame:
    """
    Computes ale : difference in the prediction when we replace the discretized feature
    with the upper and lower limit of the bins. We only replace observations within the bins.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature : FeatureDiscretizer
        Discretized representation of the feature
    predict_function : Callable
        Function to compute predictions `predict_function(data)` must work
    add_mean: bool
        Whether to add the predicted mean to ALE.

    Returns
    -------
    predictions : pd.DataFrame
        Dataframe of results:
        - rows are the same as in `data`
        - columns are bins of `feature`
    agg_predictions : [pd.DataFrame]
        DataFrame of agg results:
        - rows are mean agg
        -columns are bins of 'feature'
    """

    le = []
    for width, center in zip(feature.widths, feature.centers):
        data_bin = data[
            (
                (data[feature.name] > (center - (width / 2)))
                & (data[feature.name] <= (center + (width / 2)))
            )
        ]
        if data_bin.empty:
            le.append(0.0)
            continue
        else:
            predict_right = predict_function(
                generate_fake_data(data_bin, feature.name, center + (width / 2))
            )
            predict_left = predict_function(
                generate_fake_data(data_bin, feature.name, center - (width / 2))
            )
            le.append(pd.Series(np.subtract(predict_right, predict_left)).agg("mean"))

    ale = np.cumsum(le)
    centered_ale = ale - np.mean(ale)
    if add_mean:
        centered_ale += np.mean(predict_function(data))
    return pd.DataFrame(
        centered_ale,
        index=feature.categorical_feature.categories,
    ).T


def compute_ice_model_predictions(
    data: pd.DataFrame, feature: FeatureDiscretizer, predict_function: Callable
) -> pd.DataFrame:
    """
    Computes predictions on full data for each possible value of discretized feature.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature : FeatureDiscretizer
        Discretized representation of the feature
    predict_function : Callable
        Function to compute predictions `predict_function(data)` must work

    Returns
    -------
    predictions : pd.DataFrame
        Dataframe of results:
        - rows are the same as in `data`
        - columns are bins of `feature`
    """
    return pd.DataFrame(
        [
            predict_function(generate_fake_data(data, feature.name, center))
            for center in feature.centers
        ],
        index=feature.categorical_feature.categories,
    ).T


def aggregate_series(
    feature: FeatureDiscretizer, series: pd.Series, aggfunc: Union[str, Callable]
) -> pd.Series:
    """
    Aggregates series according to discretized feature, returns a series of aggregated values.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature
    series : pd.Series, optional
        Series containing target values
    aggfunc : Union[str, Callable]
        Aggregation function

    Returns
    -------
    agg_targets : pd.Series
        Aggregated target values, indices are bins of `feature`
    """
    return series.groupby(feature.categorical_feature).agg(aggfunc)


def compute_model_ale_results_2D(
    data: pd.DataFrame,
    feature_x: FeatureDiscretizer,
    feature_y: FeatureDiscretizer,
    predict_function: Callable,
) -> pd.DataFrame:
    """
    Computes and aggregates predictions on full data for each possible value of
    discretized features.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature_x : FeatureDiscretizer
        Discretized representation of the x feature (will be used for table columns)
    feature_y : FeatureDiscretizer
        Discretized representation of the y feature (will be used for table index)
    predict_function : Callable
        Function to compute predictions `predict_function(data)` must work

    Returns
    -------
    predictions : pd.DataFrame
        Dataframe of aggregated predictions:
        - rows are bins of feature y
        - columns are bins of feature x
    """
    le = []
    size = []

    for width_y, center_y in zip(feature_y.widths, feature_y.centers):
        le_x = []
        size_x = []
        for width_x, center_x in zip(feature_x.widths, feature_x.centers):
            max_x = center_x + (width_x / 2)
            min_x = center_x - (width_x / 2)
            max_y = center_y + (width_y / 2)
            min_y = center_y - (width_y / 2)
            data_bin = data[
                (
                    ((data[feature_x.name] > min_x) & (data[feature_x.name] <= max_x))
                    & ((data[feature_y.name] > min_y) & (data[feature_y.name] <= max_y))
                )
            ]

            if data_bin.empty:
                le_x.append(0.0)
                size_x.append(0)
                continue
            else:
                max_x_max_y = predict_function(
                    generate_fake_data(
                        generate_fake_data(data_bin, feature_x.name, max_x),
                        feature_y.name,
                        max_y,
                    )
                )

                min_x_min_y = predict_function(
                    generate_fake_data(
                        generate_fake_data(data_bin, feature_x.name, min_x),
                        feature_y.name,
                        min_y,
                    )
                )

                max_x_min_y = predict_function(
                    generate_fake_data(
                        generate_fake_data(data_bin, feature_x.name, max_x),
                        feature_y.name,
                        min_y,
                    )
                )

                min_x_max_y = predict_function(
                    generate_fake_data(
                        generate_fake_data(data_bin, feature_x.name, min_x),
                        feature_y.name,
                        max_y,
                    )
                )

                le_x.append(
                    pd.Series(
                        np.subtract(
                            np.subtract(max_x_max_y, min_x_max_y),
                            np.subtract(max_x_min_y, min_x_min_y),
                        )
                    ).agg("mean")
                )
                size_x.append(len(data_bin))
        le.append(le_x)
        size.append(size_x)

    # compute accumulated local effects
    size = np.array(size)
    le = np.array(le)
    ale = np.zeros_like(le)
    for i in range(ale.shape[0]):
        for j in range(ale.shape[1]):
            ale[i, j] = np.sum(le[: i + 1, : j + 1])

    # remove single variable effect
    ale_x_prev_index = np.c_[np.zeros((ale.shape[0], 1)), ale[:, :-1]]
    x_effect = np.cumsum(
        np.sum((ale - ale_x_prev_index) * size, axis=0) / np.sum(size, axis=0)
    )

    ale_y_prev_index = np.r_[np.zeros((1, ale.shape[1])), ale[:-1, :]]
    y_effect = np.expand_dims(
        np.cumsum(
            np.sum((ale - ale_y_prev_index) * size, axis=1, keepdims=True)
            / np.sum(size, axis=1, keepdims=True)
        ),
        axis=1,
    )

    ale_wo_single_var_effect = ale - x_effect - y_effect

    # center result to remove mean
    ale_wo_single_var_effect_centered = ale_wo_single_var_effect - np.nanmean(
        ale_wo_single_var_effect
    )

    return pd.DataFrame(
        ale_wo_single_var_effect_centered,
        index=feature_y.categorical_feature.categories,
        columns=feature_x.categorical_feature.categories,
    )


def compute_ice_model_results_2D(
    data: pd.DataFrame,
    feature_x: FeatureDiscretizer,
    feature_y: FeatureDiscretizer,
    predict_function: Callable,
    aggfunc: Union[str, Callable],
) -> pd.DataFrame:
    """
    Computes and aggregates predictions on full data for each possible value of
    discretized features.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature_x : FeatureDiscretizer
        Discretized representation of the x feature (will be used for table columns)
    feature_y : FeatureDiscretizer
        Discretized representation of the y feature (will be used for table index)
    predict_function : Callable
        Function to compute predictions `predict_function(data)` must work
    aggfunc : Union[str, Callable]
        Aggregation function

    Returns
    -------
        predictions : pd.DataFrame
            Dataframe of aggregated predictions:
            - rows are bins of feature y
            - columns are bins of feature x
    """
    comb_y = []
    for center_y in feature_y.centers:
        comb_x = []
        for center_x in feature_x.centers:
            comb_x.append(
                (
                    pd.Series(
                        predict_function(
                            generate_fake_data(
                                generate_fake_data(data, feature_x.name, center_x),
                                feature_y.name,
                                center_y,
                            )
                        )
                    )
                ).agg(aggfunc)
            )
        comb_y.append(comb_x)
    return pd.DataFrame(
        comb_y,
        index=feature_y.categorical_feature.categories,
        columns=feature_x.categorical_feature.categories,
    )


def pivot_dataframe(
    data: pd.DataFrame,
    values_name: str,
    feature_x: FeatureDiscretizer,
    feature_y: FeatureDiscretizer,
    aggfunc: Union[str, Callable],
    fill_value: Any = None,
) -> pd.DataFrame:
    """
    Wrapper around pandas pivot_table for use with discretized features.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to pivot
    values_name : str
        Name of the column of data to aggregate
    feature_x : FeatureDiscretizer
        Discretized representation of the x feature (will be used for table columns)
    feature_y : FeatureDiscretizer
        Discretized representation of the y feature (will be used for table index)
    aggfunc : Union[str, Callable]
        Aggregation function
    fill_value : Any
        Value used to fill NaN values

    Returns
    -------
    predictions : pd.DataFrame
        Pivot table of aggregated values:
        - rows are bins of feature y
        - columns are bins of feature x
    """
    return pd.pivot_table(
        data,
        values=values_name,
        index=pd.Series(feature_y.categorical_feature),
        columns=pd.Series(feature_x.categorical_feature),
        aggfunc=aggfunc,
        dropna=False,
        fill_value=fill_value,
    )


def generate_fake_data(
    data: pd.DataFrame, feature_name: str, value: Any
) -> pd.DataFrame:
    """
    Returns full dataframe except studied feature takes a constant given value.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    feature_name : str
        Name of column in `data` to change
    value : Any
        Constant value to give to column

    Returns
    -------
    data : pd.DataFrame
        Fake data with constant column
    """
    return data.assign(**{feature_name: value})


def sample_kmeans(
    data: pd.DataFrame, nb_rows: int
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Applies KMeans clustering on rows of dataframe to keep nb_rows rows.
    Returns cluster centers, number of examples in each cluster,
    and names and colors convenient for ice plots.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    nb_rows : int
        Number of rows to return (number of clusters)

    Returns
    -------
    samples : pd.DataFrame
        Cluster centers as a dataframe of shape (len(data), nb_rows)
    counts : np.ndarray
        Number of examples in each cluster, array of shape (nb_rows)
    names : List[str]
        Description of each row of samples
    colors : List[str]
        Colors for plotting each row of samples
    """
    kmeans = KMeans(
        n_clusters=nb_rows,
        tol=1e-3,
        n_init=5,
        max_iter=100,
        random_state=options.random_state,
    )
    kmeans.fit(data)
    samples = pd.DataFrame(kmeans.cluster_centers_, columns=data.columns)
    counts = np.bincount(kmeans.labels_)
    names = ["{} exemples".format(i) for i in counts]
    colors = [options.predictions_color for _ in samples]
    return samples, counts, names, colors


def sample_quantiles(
    data: pd.DataFrame, nb_rows: int
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Cuts rows of dataframe in quantiles to get nb_rows rows.
    Returns quantiles limits, number of examples in each quantile (constant),
    and names and colors convenient for ice plots.
    Specific (fake) count and color for median line.

    Parameters
    ----------
    data : pd.DataFrame
        Innut dataframe
    nb_rows : int
        Number of rows to return

    Returns
    -------
    samples : pd.DataFrame
        Quantiles limits as a dataframe of shape (len(data), nb_rows)
    counts : np.ndarray
        Number of examples in each cluster, array of shape (nb_rows)
    names : List[str]
        Description of each row of samples
    colors : List[str]
        Colors for plotting each row of samples
    """
    quantiles = [i / (nb_rows - 1) for i in range(nb_rows)]
    samples = data.apply(
        lambda x: pd.Series([np.quantile(x, q) for q in quantiles]), axis=0
    )
    counts = np.array(
        [
            2 * len(data) / nb_rows if q == 0.5 else len(data) / nb_rows
            for q in quantiles
        ]
    )
    names = ["{} quantile".format(round(q, 2)) for q in quantiles]
    colors = [
        options.special_color if q == 0.5 else options.predictions_color
        for q in quantiles
    ]
    return samples, counts, names, colors
