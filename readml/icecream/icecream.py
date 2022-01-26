"""
FBDTools library - icecream package
This module contains the main class that prepares data and draws PD/ICE/ALE plots.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Sized, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly import offline

from .check_utils import check_input_in_list, conduct_full_check
from .compute_utils import (
    aggregate_series,
    compute_ale_agg_results,
    compute_ice_model_predictions,
    compute_ice_model_results_2D,
    compute_model_ale_results_2D,
    guess_model_predict_function,
    pivot_dataframe,
    sample_kmeans,
    sample_quantiles,
)
from .config import options
from .discretizer import FeatureDiscretizer
from .plot_utils import (
    plotly_ice_box,
    plotly_ice_lines,
    plotly_partial_dependency,
    plotly_partial_dependency_2d_hist,
    plotly_partial_dependency_2d_scatter,
)


class IceCream(object):
    """
    Class that generates and contains predictions and aggregations used to
    draw PDPlots, ICE plots and ALE plots.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature_names : List[str]
        List of names of columns from data to discretize and analyse
    bins : Union[Dict[str, Union[int, Sized, None]], int] = {}
        Bin definitions for features:
        - None -> all bin definitions are guessed by module
        - integer -> value is used as number of bins for all features
        - dict -> keys are features names, values are:

            - integers for number of bins (0 for categorical features)
            - None to let the module decide
            - Sized to define specific bins

        Empty dict by default (which means all bins will be guessed)

    use_ale : bool, optional
        If True, computes ALE: Accumulated Local Effects.
        Can only be used for numerical features. (the Default is False)
    model : scikit-learn model, optional
        Model to compute predictions using provided data,
        `model.predict(data)` must work
    predictions : Optional[Sized]
        Series containing predictions for rows in data,
        used if no model can be given
    targets : Optional[Sized]
        Series containing targets for rows in data
    aggfunc : str, optional
        Aggregation function for targets and predictions aggregation (the default is "mean")
    use_classif_proba : bool, optional
        If True, use prediction probability as model output,
        only used if model is a classifier. (the default is True)
    clip_quantile : float, optional
        Quantile to clip the feature values for continuous features,
        set to 0 to disable clipping (the default is 0.0)
    quantile_based : bool, optional
        Option to use a quantile-based discretization function for
        continuous features (instead of a linear discretization),  (the default is False)

    Attributes
    ----------
    features : List[FeatureDiscretizer]
        Discretized representations of the studied features
    predictions : Dict[str, pd.DataFrame]
        Dictionary of predictions, keys are feature names, values are
        dataframes of predictions for each bin
    agg_predictions : Dict[str, pd.Series]
        Dictionary of aggregated predictions values, keys are feature names
    agg_targets : Dict[str, pd.Series]
        Dictionary of aggregated target values, keys are feature names
    - samples : Dict[str, pd.DataFrame]
        Dict of dataframes of computed predictions samples/clusters
        if ice line plot is drawn. Not filled until `draw` method is called.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_names: List[str],
        bins: Union[Dict[str, Union[int, Sized, None]], int] = {},
        use_ale: bool = False,
        model: Optional[Any] = None,
        predictions: Optional[Sized] = None,
        targets: Optional[Sized] = None,
        aggfunc: str = "mean",
        use_classif_proba: bool = True,
        clip_quantile: float = 0.0,
        quantile_based: bool = False,
        class_name=None,
    ) -> None:
        """
        Creates class instance with attributes, then run results computation.
        """
        conduct_full_check(
            data,
            feature_names,
            bins,
            model,
            predictions,
            targets,
            aggfunc,
            ["mean", "median"],
            use_classif_proba,
            clip_quantile,
            use_ale,
        )

        # transform bins if a scalar was given
        if isinstance(bins, (int, np.integer)):
            bins = {name: bins for name in feature_names}

        self.features = [
            FeatureDiscretizer(
                data[name], bins.get(name), clip_quantile, quantile_based
            )
            for name in feature_names
        ]
        self.predictions = dict()  # type: Dict[str, Any]
        self.agg_predictions = dict()  # type: Dict[str, Any]
        self.agg_targets = dict()  # type: Dict[str, Any]
        self.samples = dict()  # type: Dict[str, Any]
        self._aggfunc = aggfunc
        self.use_ale = use_ale
        self.class_name = class_name
        self._run(data, model, predictions, targets, use_classif_proba, use_ale)

    def __repr__(self) -> str:
        return "{}: features ({})".format(
            self.__class__.__name__,
            ", ".join([feature.name for feature in self.features]),
        )

    def _run(
        self,
        data: pd.DataFrame,
        model: Any,
        predictions: Optional[Sized],
        targets: Optional[Sized],
        use_classif_proba: bool,
        use_ale: bool,
    ) -> None:
        """
        Run all operations to compute data used in PDPlots, ICE plots and ALE plots.
        Operations are run only if model/targets/predictions are given.
        """
        if predictions is not None:
            predictions = pd.Series(predictions)
            self.agg_predictions = {
                feature.name: aggregate_series(feature, predictions, self._aggfunc)
                for feature in self.features
            }

        if model is not None:
            predict_function = guess_model_predict_function(
                model, use_classif_proba, self.class_name
            )

            if use_ale:
                for feature in self.features:
                    self.agg_predictions[feature.name] = compute_ale_agg_results(
                        data, feature, predict_function
                    )

            else:
                self.predictions = {
                    feature.name: compute_ice_model_predictions(
                        data, feature, predict_function
                    )
                    for feature in self.features
                }
                self.agg_predictions = {
                    name: self.predictions[name].agg(self._aggfunc, axis=0)
                    for name in self.predictions
                }

        if targets is not None:
            targets = pd.Series(targets)
            self.agg_targets = {
                feature.name: aggregate_series(feature, targets, self._aggfunc)
                for feature in self.features
            }

    def draw(
        self,
        kind: str = "pdp",
        show: bool = True,
        save_path: Optional[str] = None,
        ice_nb_lines: int = 15,
        ice_clustering_method: str = "quantiles",
    ) -> Dict[str, go.FigureWidget]:
        """
        Builds plots, optionally shows them in current notebook and save them in HTML format.

        Parameters
        ----------
        kind : str, optional
            Kind of plot to draw, possibilities are:
            - "pdp": draws a Partial Dependency Plot
            - "box": draws a box plot of predictions for each bin of features
            - "ice": draws a Individual Conditional Expectation plot
            - "ale": draws an Accumulated Local Effects plot
            (the default is "pdp")

        show : bool, optional
            Option to show the plots in notebook (the default is True)
        save_path : Optional[str]
            Path to directory to save the plots,
            directory is created if it does not exist
        ice_nb_lines : int, optional
            Number of lines to draw if kind="ice" (the default is 15)
        ice_clustering_method : str, optional
            Sampling or clustering method to compute the best lines to draw if kind="ice",
            available methods:
            - "kmeans": automatic clustering using KMeans to get representative lines
            - "quantiles": division of predictions in quantiles to get lines
            - "random": random selection of rows among predictions
            (the default is "quantiles")

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            Dictionary of generated plots,
            keys are feature names, values are Plotly objects
        """
        check_input_in_list(kind, ["pdp", "box", "ice", "ale"])
        check_input_in_list(ice_clustering_method, ["kmeans", "quantiles", "random"])

        # specific arguments for ice plot method
        if kind == "ice":
            figures = self._ice_lines_plot(
                nb_lines=ice_nb_lines, clustering_method=ice_clustering_method
            )
        elif kind == "box":
            figures = self._ice_box_plot()
        elif kind == "pdp" or kind == "ale":
            figures = self._line_plot()

        if save_path is not None:
            save_dir = os.path.abspath(save_path)
            os.makedirs(save_dir, exist_ok=True)
            for name, figure in figures.items():
                filename = os.path.join(save_dir, "{}_{}.html".format(kind, name))
                offline.plot(figure, filename=filename, auto_open=False)

        if show:
            # for loop on features list for sorted show
            for feature in self.features:
                if feature.name in figures:
                    offline.iplot(figures[feature.name])
        return figures

    def _line_plot(self) -> Dict[str, go.FigureWidget]:
        """
        Returns a dict of N Plotly line plots for the N features contained in instance.

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            Dictionary of generated plots,
            keys are feature names, values are Plotly objects
        """
        if not self.predictions and self.agg_predictions and not self.use_ale:
            warnings.warn(
                "No model was provided, shown predictions were aggregated"
                " and thus do not explain the model that produced them"
            )
        figures = dict()
        for feature in self.features:
            name = feature.name
            figures[name] = plotly_partial_dependency(
                feature,
                self.agg_predictions.get(name),
                self.agg_targets.get(name),
                self._aggfunc,
                self.use_ale,
            )
        return figures

    def _ice_box_plot(self) -> Dict[str, go.FigureWidget]:
        """
        Returns a dict of N Plotly ICE Box plots for the N features in instance.

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            Dictionary of generated plots,
            keys are feature names, values are Plotly objects
        """
        if not self.predictions and self.agg_predictions:
            warnings.warn(
                "No model was provided, predictions cannot be shown on the ICE box plot"
            )
        figures = dict()
        for feature in self.features:
            name = feature.name
            figures[name] = plotly_ice_box(
                feature,
                self.predictions.get(name),
                self.agg_targets.get(name),
                self._aggfunc,
            )
        return figures

    def _ice_lines_plot(
        self, nb_lines: int, clustering_method: str
    ) -> Dict[str, go.FigureWidget]:
        """
        Returns a dict of N Plotly ICE plots for the N features in instance.

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            Dictionary of generated plots,
            keys are feature names, values are Plotly objects
        """
        if not self.predictions and self.agg_predictions:
            warnings.warn(
                "No model was provided, predictions cannot be shown on the ICE plot"
            )
        if nb_lines < 2:
            raise ValueError(
                "Number of lines for ICE plot must be greater than 1"
                ", use PDP to show a 1 line aggregation plot"
            )

        figures = dict()
        for feature in self.features:
            name = feature.name
            predictions = self.predictions.get(name)
            if predictions is not None:
                assert (
                    len(predictions) >= nb_lines
                ), "Number of lines must be inferior or equal to length of dataset"

                if clustering_method == "random":
                    samples = predictions.sample(n=nb_lines)
                    counts = np.full(nb_lines, len(predictions) / nb_lines)
                    names = ["" for _ in samples]
                    colors = [options.predictions_color for _ in samples]
                elif clustering_method == "kmeans":
                    samples, counts, names, colors = sample_kmeans(
                        predictions, nb_lines
                    )
                elif clustering_method == "quantiles":
                    samples, counts, names, colors = sample_quantiles(
                        predictions, nb_lines
                    )

                figures[name] = plotly_ice_lines(
                    feature,
                    samples,
                    counts,
                    names,
                    colors,
                    self.agg_targets.get(name),
                    self._aggfunc,
                )
                self.samples[name] = samples
        return figures


class IceCream2D(object):
    """
    Class that generates and contains predictions and aggregations used to
    draw 2D interaction plots (partial dependencies or ALE heatmaps).

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe of model inputs
    feature_x : str
        Name of column from data to discretize for x axis
    feature_y : str
        Name of column from data to discretize for y axis
    bins_x : Optional[Union[int, Sized]]
        Bin definition for feature_x
        - None -> bin definition is guessed by module
        - integer -> value is used as number of bins for all features
        - Sized -> define specific bins
    bins_y : Optional[Union[int, Sized]]
        Bin definition for feature_y
        - None -> bin definition is guessed by module
        - integer -> value is used as number of bins for all features
        - Sized -> define specific bins
    use_ale : bool, optional
        If True, computes ALE: Accumulated Local Effects.
        Can only be used for numerical features. (the default is False)
    model : scikit-learn model, optional
        Model to compute predictions using provided data,
        `model.predict(data)` must work
    predictions : Optional[Sized]
        Series containing predictions for rows in data,
        used if no model can be given
    targets : Optional[Sized]
        Series containing targets for rows in data
    aggfunc : str, optional
        Aggregation function for targets and predictions aggregation
        (the default is "mean")
    use_classif_proba : bool
        If True, use prediction probability as model output,
        only used if model is a classifier
        (the default is True)
    clip_quantile : float, optional
        Quantile to clip the feature values for continuous features,
        set to 0 to disable clipping
        (the default is 0.0)
    quantile_based : bool, optional
        Option to use a quantile-based discretization function for
        continuous features (instead of a linear discretization).
        (the default is False)

    Attributes
    ----------
    feature_x : FeatureDiscretizer
        Discretized representation of feature for x axis
    feature_y : FeatureDiscretizer
        Discretized representation of feature for y axis
    counts : List[pd.DataFrame]
        List of counts of number of rows for each square of heatmap
    agg_predictions : pd.DataFrame
        Dataframe of aggregated predictions for each bin of features x and y
    agg_targets : pd.DataFrame
        Dataframe of aggregated targets for each bin of features x and y
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_x: str,
        feature_y: str,
        bins_x: Optional[Union[int, Sized]],
        bins_y: Optional[Union[int, Sized]],
        use_ale: bool = False,
        model: Optional[Any] = None,
        predictions: Optional[Sized] = None,
        targets: Optional[Sized] = None,
        aggfunc: str = "mean",
        use_classif_proba: bool = True,
        clip_quantile: float = 0.0,
        quantile_based: bool = False,
    ) -> None:
        """
        Creates class instance with attributes, then run results computation.
        """
        conduct_full_check(
            data,
            [feature_x, feature_y],
            {"x": bins_x, "y": bins_y},
            model,
            predictions,
            targets,
            aggfunc,
            ["mean", "median"],
            use_classif_proba,
            clip_quantile,
            use_ale,
        )

        self.feature_x = FeatureDiscretizer(
            data[feature_x], bins_x, clip_quantile, quantile_based
        )
        self.feature_y = FeatureDiscretizer(
            data[feature_y], bins_y, clip_quantile, quantile_based
        )
        self.counts = None
        self.use_ale = use_ale
        self.agg_predictions = None
        self.agg_targets = None
        self._aggfunc = aggfunc
        self._run(data, model, predictions, targets, use_classif_proba, use_ale)

    def __repr__(self) -> str:
        return "{}: feature_x ({}), feature_y ({})".format(
            self.__class__.__name__, self.feature_x.name, self.feature_y.name
        )

    def _run(
        self,
        data: pd.DataFrame,
        model: Any,
        predictions: Optional[Sized],
        targets: Optional[Sized],
        use_classif_proba: bool,
        use_ale: bool,
    ) -> None:
        """
        Run all operations to compute data used in heatmaps.
        """
        # fake data for count pivot table
        data_temp = data.assign(values=0)
        self.counts = pivot_dataframe(
            data_temp, "values", self.feature_x, self.feature_y, "count", fill_value=0
        )

        if predictions is not None:
            predictions = pd.Series(predictions)
            data_temp = data.assign(values=predictions)
            self.agg_predictions = pivot_dataframe(
                data_temp, "values", self.feature_x, self.feature_y, self._aggfunc
            )

        if model is not None:
            predict_function = guess_model_predict_function(model, use_classif_proba)

            if use_ale:
                self.agg_predictions = compute_model_ale_results_2D(
                    data, self.feature_x, self.feature_y, predict_function
                )

            else:
                self.agg_predictions = compute_ice_model_results_2D(
                    data,
                    self.feature_x,
                    self.feature_y,
                    predict_function,
                    self._aggfunc,
                )

        if targets is not None:
            targets = pd.Series(targets)
            data_temp = data.assign(values=targets)
            self.agg_targets = pivot_dataframe(
                data_temp, "values", self.feature_x, self.feature_y, self._aggfunc
            )

    def draw(
        self, kind: str = "hist", show: bool = True, save_path: Optional[str] = None
    ) -> List[go.FigureWidget]:
        """
        Builds plots, optionally shows them in current notebook and save them in HTML format.

        Parameters
        ----------
        show : bool, optional
            Option to show the plots in notebook
            (the default is True)
        save_path : Optional[str]
            Path to directory to save the plots,
            directory is created if it does not exist
            (the default is None)
        kind : str
            Kind of plot to draw, possibilities are:
            - "hist": histograms for feature values, heatmap for predictions and targets
            - "scatter": scatter for feature values, heatmap for predictions and targets
            (the default is "hist")

        Returns
        -------
        figures : List[go.FigureWidget]
            List of generated plots,
            1 plot for predictions if model or predictions were given
            1 plot for targets if they were given
        """
        check_input_in_list(kind, ["hist", "scatter"])
        if kind == "hist":
            figures = self._pd_hist_plot()
        elif kind == "scatter":
            figures = self._pd_scatter_plot()

        if save_path is not None:
            save_dir = os.path.abspath(save_path)
            os.makedirs(save_dir, exist_ok=True)
            for i, figure in enumerate(figures):
                filename = os.path.join(save_dir, "heatmap_{}.html".format(i))
                offline.plot(figure, filename=filename, auto_open=False)

        if show:
            # for loop on features list for sorted show
            for figure in figures:
                offline.iplot(figure)
        return figures

    def _pd_hist_plot(self) -> List[go.FigureWidget]:
        """
        Returns a list of Plotly PDP 2D plots.

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            List of generated plots,
            1 plot for predictions if model or predictions were given
            1 plot for targets if they were given
        """
        figures = []  # type: List[go.FigureWidget]
        if self.agg_predictions is not None:
            figures.append(
                plotly_partial_dependency_2d_hist(
                    self.feature_x,
                    self.feature_y,
                    self.counts,
                    self.agg_predictions,
                    "predictions",
                )
            )
        if self.agg_targets is not None:
            figures.append(
                plotly_partial_dependency_2d_hist(
                    self.feature_x,
                    self.feature_y,
                    self.counts,
                    self.agg_targets,
                    "targets",
                )
            )
        return figures

    def _pd_scatter_plot(self) -> List[go.FigureWidget]:
        """
        Returns a list of Plotly PDP 2D plots.

        Returns
        -------
        figures : Dict[str, go.FigureWidget]
            List of generated plots,
            1 plot for predictions if model or predictions were given
            1 plot for targets if they were given
        """
        figures = []  # type: List[go.FigureWidget]
        if self.agg_predictions is not None:
            figures.append(
                plotly_partial_dependency_2d_scatter(
                    self.feature_x,
                    self.feature_y,
                    self.counts,
                    self.agg_predictions,
                    "predictions",
                )
            )
        if self.agg_targets is not None:
            figures.append(
                plotly_partial_dependency_2d_scatter(
                    self.feature_x,
                    self.feature_y,
                    self.counts,
                    self.agg_targets,
                    "targets",
                )
            )
        return figures
