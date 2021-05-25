"""
FBDTools library - icecream package
This module helps discretize features for icecream.
"""

import warnings
from typing import Sized, Tuple, Union

import numpy as np
import pandas as pd

from .check_utils import check_clip_quantile
from .config import options


class FeatureDiscretizer(object):
    """
    Class that discretizes the values of a feature and contains all properties
    needed to compute PD/ICE/ALE aggregations.

    - If feature is continuous, creates bins.
    - If feature is categorical, uses feature values as discret bins.

    Parameters
    ----------
    series : pd.Series
        Series of values of feature to discretize
    bins : Union[Sized, int, None]
        Bins definition:
        - integer for number of bins (0 if feature is categorical)
        - None to let the module decide using series values
        - Sized to define specific bins
        (the default is None)
    clip_quantile : float
        Quantile to clip the feature values for continuous features,
        set to 0 to disable clipping (the default is 0.0)
    quantile_based : bool
        Option to use a quantile-based discretization function for
        continuous features (instead of a linear discretization),
        False by default

    Attributes
    ----------
    name: str
        Name of the discretized feature
    centers : pd.Index
        Centers of the discrete bins
        (equals to ordered values if feature is categorical)
    widths : Optional[pd.Float64Index]
        Widths of the bins (None if feature is categorical)
    counts : pd.Series
        Number of exemples from series for each bin
        (height of the histogram/bar plot)
    categorical_feature : pd.Categorical
        Categorical view of the discretized feature
    """

    def __init__(
        self,
        series: pd.Series,
        bins: Union[Sized, int, None] = None,
        clip_quantile: float = 0.0,
        quantile_based: bool = False,
    ) -> None:
        """
        Conducts full discretization to defines all class attributes.
        """
        self.name = series.name
        if bins is None:
            bins = self._guess_bins(series)
        if isinstance(bins, (int, np.integer)) and bins == 0:
            # feature is categorical
            self.categorical_feature = pd.Categorical(series)
            self.widths = None
            self.centers = self.categorical_feature.categories
        else:
            # feature is continuous
            clipped_series = self._clip_values(series, clip_quantile)
            categorical_feature, _ = self._discretize_values(
                clipped_series, bins, quantile_based
            )
            self.categorical_feature = pd.Categorical(categorical_feature)
            self.widths = (
                self.categorical_feature.categories.right
                - self.categorical_feature.categories.left
            )
            self.centers = self.categorical_feature.categories.mid
        self.counts = self.categorical_feature.value_counts()

    def __len__(self) -> int:
        return len(self.centers)

    def __repr__(self) -> str:
        return "{}: '{}' discretized in {} bins".format(
            self.__class__.__name__, self.name, len(self)
        )

    def _discretize_values(
        self,
        series: pd.Series,
        bins: Union[Sized, int, None] = None,
        quantile_based: bool = False,
    ) -> Tuple[pd.Series, np.ndarray]:
        """
        Discretizes series and returns categorical view of the series and precise
        bin boundaries.

        Parameters
        ----------
        series : pd.Series
            Series of values
        bins : Union[Sized, int, None]
            Bins definition:
            - integer for number of bins (0 if feature is categorical)
            - None to let the module decide using series values
            - Sized to define specific bins
            (the default is None)
        quantile_based : bool
            Option to use a quantile-based discretization function for
            continuous features (instead of a linear discretization),
            False by default

        Returns
        -------
        categorical_feature : pd.Series
            Categorical view of the discretized feature
        ret_bins : np.ndarray
            Bin boundaries
        """
        if quantile_based:
            categorical_feature, ret_bins = pd.qcut(
                series, q=bins, retbins=True, duplicates="drop"
            )
        else:
            categorical_feature, ret_bins = pd.cut(series, bins=bins, retbins=True)

        if len(ret_bins) > options.max_recommended_categories:
            warnings.warn(
                (
                    "Feature {} is discretized in {} bins"
                    ", which is over the recommended maximum of {} bins"
                ).format(self.name, len(ret_bins), options.max_recommended_categories)
            )
        return categorical_feature, ret_bins

    @staticmethod
    def _guess_bins(series: pd.Series) -> int:
        """
        Returns number of bins by guessing if series is categorical or continuous.
        Feature is categorical if 1 of these conditions is fulfilled:

            - series dtype is categorical or object (string)
            - series is empty
            - number of unique values of series is below N and below X percent of
              series length

        Parameters
        ----------
        series : pd.Series
            Series of values of feature to discretize

        Returns
        -------
        bins : int
            Number of bins
        """
        if (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or (len(series) == 0)
            or (len(series) <= options.default_number_bins)
        ):
            return 0
        number_uniques = series.nunique()
        unicity_ratio = number_uniques / len(series)
        if (unicity_ratio <= options.max_unicity_ratio) and (
            number_uniques <= options.max_categories
        ):
            return 0
        return options.default_number_bins

    @staticmethod
    def _clip_values(series: pd.Series, clip_quantile: float) -> pd.Series:
        """
        Clip extreme values of series using quantiles values as boundaries.

        Parameters
        ----------
        series : pd.Series
            Series of values
        clip_quantile : float
            Value of lower quantile to clip,
            will be used for upper quantile too (upper = 1 - lower)

        Returns
        -------
            series : pd.Series
                Series of clipped values
        """
        check_clip_quantile(clip_quantile)
        return series.clip(
            series.quantile(clip_quantile), series.quantile(1 - clip_quantile)
        )
