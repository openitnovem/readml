# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

"""
FBDTools library - icecream package
This module contains helper methods that generate plot objects useful for
drawing PD/ICE/ALE plots using data objects containing features,
targets and predictions
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .config import options
from .discretizer import FeatureDiscretizer

FEATURE_CONST = "feature "


def detect_axis_range(*args: Union[pd.Series, pd.DataFrame]) -> Optional[List[float]]:
    """
    Determines axis range based on content of data.
    Returns [-0.05, 1.05] if data is contained in this interval, else None.

    Parameters
    ----------
    args : Union[pd.Series, pd.DataFrame]
        Dataframes or series containing values shown on axis,
        each arg should have an attribute `values` that returns a np.ndarray

    Returns
    -------
    range : Optional[List[float]]
        Axis range values, None to let the graph library decide
    """
    # prevent problems if data is not numerical
    try:
        full_data = np.vstack([x.values for x in args if hasattr(x, "values")])
        if (np.nanmin(full_data) >= 0) and (np.nanmax(full_data) <= 1):
            return [-0.05, 1.05]
    except TypeError:
        return None
    except ValueError:
        return None
    except AttributeError:
        return None
    return None


def plotly_background_bars(
    feature: FeatureDiscretizer,
    opacity: float = 0.5,
    marker: Dict[str, Any] = dict(
        color=options.bars_color, line=dict(color=options.bars_color, width=1)
    ),
    name: str = "histogram",
    hbar: bool = False,
) -> go.Bar:
    """
    Generates a background bar/histogram plot with Plotly using discretized
    representation of the feature.
    Wrapper around go.Bar with sane defaults with present usage.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature
    opacity : float
        Opacity of the bars
    marker : Dict[str, Any], optional
        Plotly option: marker definition (the default is dict(size=10, line=dict(width=1)))
    name : str, optional
        Plotly option: Name given to the trace (the default is "output")
    hbar : bool, optional
        Option to plot bars horizontally (the default is False)

    Returns
    -------
    bar : go.Bar
        Bar plot representing the distribution of values of feature
    """
    if hbar:
        return go.Bar(
            x=feature.counts,
            y=feature.centers,
            width=feature.widths,
            opacity=opacity,
            marker=marker,
            xaxis="x2",
            orientation="h",
            name=name,
        )
    else:
        return go.Bar(
            x=feature.centers,
            y=feature.counts,
            width=feature.widths,
            opacity=opacity,
            marker=marker,
            yaxis="y2",
            name=name,
        )


def plotly_line(
    feature: FeatureDiscretizer,
    outputs: pd.Series,
    mode: str = "lines+markers",
    marker: Dict[str, Any] = dict(size=10, line=dict(width=1)),
    line: Optional[Dict[str, Any]] = None,
    name: Optional[str] = "output",
    showlegend: bool = True,
) -> go.Scatter:
    """
    Generates a line plot with Plotly to show outputs for discretized feature.
    Wrapper around go.Scatter with sane defaults with present usage.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    outputs : pd.Series
        Y values
    mode : str, optional
        Plotly option: line plot mode (the default is "lines+markers")
    marker : Dict[str, Any], optional
        Plotly option: marker definition (the default is dict(size=10, line=dict(width=1)))
    line : Dict[str, Any], optional None
        Plotly option: line definition, mainly useful if mode="lines"
        (the default is None)
    name : str, optional
        Plotly option: Name given to the trace (the default is "output")
    showlegend : bool, optional
        Plotly option: Option to show name in legend or not (the default is True)

    Returns
    -------
    line : go.Scatter
        Line plot representing the output values
    """
    return go.Scatter(
        x=feature.centers,
        y=outputs,
        mode=mode,
        marker=marker,
        line=line,
        name=name,
        showlegend=showlegend,
    )


def plotly_boxes(
    outputs: pd.Series, marker: Optional[Dict[str, Any]] = None, name: str = "output"
) -> go.Box:
    """
    Generates boxes with Plotly using 2D dataframe.
    Uses dataframe columns as x values for boxes, and rows generate the boxes.

    Parameters
    ----------
    outputs : pd.Series
        Dataframe where columns are bins of discretized feature,
        rows are predictions
    name : str, optional
        Name given to the boxes in the legend (the default is "output")

    Returns
    -------
    boxes : go.Box
        Boxes plot representing the distribution of values
        of rows of outputs for each column of outputs
    """
    melted = outputs.melt()
    return go.Box(x=melted.variable, y=melted.value, marker=marker, name=name)


def plotly_partial_dependency(
    feature: FeatureDiscretizer,
    agg_predictions: Optional[pd.Series],
    agg_targets: Optional[pd.Series],
    aggfunc: str = "",
    use_ale: bool = False,
) -> go.FigureWidget:
    """
    Generates a Partial Dependency Plot for given feature, predictions and targets.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    agg_predictions : Optional[pd.Series])
        Aggregated predictions for feature, used for y values
    agg_targets : Optional[pd.Series]
        Aggregated values of targets, used for y values
    aggfunc : str = ""
        Name of aggregation function for legend
    use_ale : bool, optional
        True if use ale else False (the default is False)

    Returns
    -------
    figure : go.FigureWidget
        Full partial dependency plot
    """
    data = [plotly_background_bars(feature)]
    if use_ale:
        name = "ALE"
        multi_name = "ALE {}"
    else:
        name = "{} effect".format(aggfunc)
        multi_name = "effect {}"

    if agg_predictions is not None:
        # in case there are several prediction values (multiclass)
        if len(agg_predictions.shape) > 1:
            for name, values in agg_predictions.iterrows():
                data.append(
                    plotly_line(feature, values, name=multi_name.format(aggfunc, name))
                )
        else:
            data.append(plotly_line(feature, agg_predictions, name=name))

    if agg_targets is not None and not use_ale:
        if len(agg_targets.shape) > 1:
            for name, values in agg_targets.iterrows():
                data.append(
                    plotly_line(
                        feature, values, name="{} target {}".format(aggfunc, name)
                    )
                )
        else:
            data.append(
                plotly_line(
                    feature,
                    agg_targets,
                    name="{} target".format(aggfunc),
                    marker=dict(
                        size=10, color=options.targets_color, line=dict(width=1)
                    ),
                )
            )

    yaxis_range = detect_axis_range(agg_predictions, agg_targets)

    layout = go.Layout(
        xaxis=dict(title=FEATURE_CONST + feature.name),
        yaxis=dict(range=yaxis_range, title="value", overlaying="y2"),
        yaxis2=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
        ),
    )
    return go.FigureWidget(data=data, layout=layout)


def plotly_ice_box(
    feature: FeatureDiscretizer,
    predictions: Optional[pd.DataFrame],
    agg_targets: Optional[pd.Series],
    aggfunc: str = "",
) -> go.FigureWidget:
    """
    Generates an ICE Box Plot with Plotly for given feature, predictions and targets.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    predictions : pd.DataFrame
        Dataframe of predictions for feature, columns must be bins of `feature`
    agg_targets : pd.Series
        Aggregated values of targets, used for y values
    aggfunc : str = ""
        Name of aggregation function for legend

    Returns
    -------
    figure : go.FigureWidget
        Full ICE Box plot
    """
    data = [plotly_background_bars(feature)]

    if predictions is not None:
        data.append(
            plotly_boxes(
                predictions,
                name="prediction",
                marker=dict(color=options.predictions_color),
            )
        )

    if agg_targets is not None:
        data.append(
            plotly_line(
                feature,
                agg_targets,
                name="{} label".format(aggfunc),
                marker=dict(size=10, color=options.targets_color, line=dict(width=1)),
            )
        )

    yaxis_range = detect_axis_range(predictions, agg_targets)

    layout = go.Layout(
        xaxis=dict(title=FEATURE_CONST + feature.name),
        yaxis=dict(range=yaxis_range, title="value", overlaying="y2"),
        yaxis2=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
        ),
    )
    return go.FigureWidget(data=data, layout=layout)


def plotly_ice_lines(
    feature: FeatureDiscretizer,
    samples: pd.DataFrame,
    counts: np.ndarray,
    names: List[str],
    colors: List[str],
    agg_targets: Optional[pd.Series],
    aggfunc: str = "",
) -> go.FigureWidget:
    """
    Generates an ICE Plot with Plotly for given feature, predictions and targets.
    Operates a sampling or clustering on predictions to draw a limited number
    of lines.

    Parameters
    ----------
    feature : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    samples : pd.DataFrame
        Quantiles limits as a dataframe of shape (len(data), nb_rows)
    counts : np.ndarray
        Number of examples in each cluster, array of shape (nb_rows)
    names : List[str]
        Description of each row of samples
    colors : List[str]
        Colors for plotting each row of samples
    agg_targets : pd.Series
        Aggregated values of targets, used for y values
    aggfunc : str = ""
        Name of aggregation function for legend

    Returns
    -------
    figure : go.FigureWidget
        Full ICE plot
    """
    data = [plotly_background_bars(feature)]

    for (_, sample), count, name, color in zip(
        samples.iterrows(), counts, names, colors
    ):
        data.append(
            plotly_line(
                feature,
                sample,
                name=name,
                showlegend=False,
                mode="lines",
                line=dict(color=color, width=count / counts.mean()),
            )
        )

    if agg_targets is not None:
        data.append(
            plotly_line(
                feature,
                agg_targets,
                name="{} label".format(aggfunc),
                marker=dict(size=10, color=options.targets_color, line=dict(width=1)),
            )
        )

    yaxis_range = detect_axis_range(samples, agg_targets)

    layout = go.Layout(
        xaxis=dict(title=FEATURE_CONST + feature.name),
        yaxis=dict(range=yaxis_range, title="value", overlaying="y2"),
        yaxis2=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            showticklabels=False,
        ),
    )
    return go.FigureWidget(data=data, layout=layout)


def plotly_partial_dependency_2d_scatter(
    feature_x: FeatureDiscretizer,
    feature_y: FeatureDiscretizer,
    counts: pd.DataFrame,
    values: pd.DataFrame,
    name: Optional[str] = "output",
) -> go.FigureWidget:
    """
    Generates a heatmap + scatter with Plotly for given features, counts and values.
    Heatmap represent predictions or target values, scatter represent histogram.

    Parameters
    ----------
    feature_x : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    feature_y : FeatureDiscretizer
        Discretized representation of the feature, used for y values
    counts : pd.DataFrame
        Pivot table of number of examples in each bin of features
    values : pd.DataFrame
        Pivot table of output values (predictions or targets) for each bin of features
    name : str
        Plotly option: Name given to the trace (the default is "output")

    Returns
    -------
    figure : go.FigureWidget
        Full heatmap + scatter plot
    """
    zaxis_range = detect_axis_range(values)
    zmin = None
    zmax = None
    if zaxis_range is not None:
        zmin = zaxis_range[0]
        zmax = zaxis_range[1]

    heatmap = go.Heatmap(
        x=feature_x.centers,
        y=feature_y.centers,
        z=values,
        zmin=zmin,
        zmax=zmax,
        colorscale=options.heatmap_colorscale,
        colorbar=dict(len=0.7),
        name=name,
    )

    mesh_x, mesh_y = np.meshgrid(feature_x.centers, feature_y.centers)
    counts_flat = counts.values.flatten()
    scatter = go.Scatter(
        x=mesh_x.flatten(),
        y=mesh_y.flatten(),
        mode="markers",
        marker=dict(
            opacity=1,
            color="rgba(0,0,0,0)",
            size=10 * counts_flat / counts_flat.max(),
            line=dict(color="black", width=2),
        ),
        name="histogram",
        text=counts_flat,
    )

    data = [heatmap, scatter]

    layout = go.Layout(
        xaxis=dict(title=FEATURE_CONST + feature_x.name, showgrid=False),
        yaxis=dict(title=FEATURE_CONST + feature_y.name, showgrid=False),
        autosize=False,
        showlegend=True,
        hovermode="closest",
        title=name,
    )
    return go.FigureWidget(data=data, layout=layout)


def plotly_partial_dependency_2d_hist(
    feature_x: FeatureDiscretizer,
    feature_y: FeatureDiscretizer,
    counts: pd.DataFrame,
    values: pd.DataFrame,
    name: Optional[str] = "output",
) -> go.FigureWidget:
    """
    Generates a heatmap + 2 histograms with Plotly for given features and values.
    Heatmap represent predictions or target values.

    Parameters
    ----------
    feature_x : FeatureDiscretizer
        Discretized representation of the feature, used for x values
    feature_y : FeatureDiscretizer
        Discretized representation of the feature, used for y values
    counts : pd.DataFrame
        Pivot table of number of examples in each bin of features
    values : pd.DataFrame
        Pivot table of output values (predictions or targets) for each bin of features
    name : str
        Plotly option: Name given to the trace (the default is "output")

    Returns
    -------
    figure : go.FigureWidget
        Full heatmap + hists plot
    """
    zaxis_range = detect_axis_range(values)
    zmin = None
    zmax = None
    if zaxis_range is not None:
        zmin = zaxis_range[0]
        zmax = zaxis_range[1]

    heatmap = go.Heatmap(
        x=feature_x.centers,
        y=feature_y.centers,
        z=values,
        zmin=zmin,
        zmax=zmax,
        colorscale=options.heatmap_colorscale,
        colorbar=dict(len=0.7),
        name=name,
        text=counts,
    )
    hist_x = plotly_background_bars(feature_x)
    hist_y = plotly_background_bars(feature_y, hbar=True)

    data = [heatmap, hist_x, hist_y]

    layout = go.Layout(
        xaxis=dict(
            title=FEATURE_CONST + feature_x.name,
            domain=[0, 0.85],
            showgrid=False,
            ticks="",
        ),
        yaxis=dict(
            title=FEATURE_CONST + feature_y.name,
            domain=[0, 0.85],
            showgrid=False,
            ticks="",
        ),
        xaxis2=dict(domain=[0.85, 1], showgrid=False, ticks="", showticklabels=False),
        yaxis2=dict(domain=[0.85, 1], showgrid=False, ticks="", showticklabels=False),
        autosize=False,
        bargap=0,
        showlegend=True,
        hovermode="closest",
        title=name,
    )
    return go.FigureWidget(data=data, layout=layout)
