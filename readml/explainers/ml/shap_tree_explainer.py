# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from readml.logger import logger


class ShapTreeExplainer:
    """
    Allows to explain globally or locally a tree based model using Tree SHAP algorithms.
    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and
    ensembles of trees.

    Attributes
    ----------
    model : scikit-learn model or Pyspark model
        A tree based model. Following models are supported:
        XGBoost, LightGBM, CatBoost, Pyspark & most tree-based models in scikit-learn...
    features_name : List[str]
        List of features names used to train the model
    classif : bool
        True, if it's a classification problem, else False
    """

    def __init__(self, model: Any, features_name: List[str], classif: bool) -> None:
        self.model = model
        self.features_name = features_name
        self.classif = classif

    def global_explainer(
        self, train_data: pd.DataFrame
    ) -> Tuple[plt.figure, plt.figure]:
        """
        Create a SHAP feature importance plot and SHAP summary plot, colored by feature
        values using Tree Explainer.

        Parameters
        ----------
        train_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model

        Returns
        -------
        shap_fig1, shap_fig2 : Tuple[plt.figure, plt.figure]
            SHAP summary plots
        """
        shap_values = shap.TreeExplainer(self.model).shap_values(
            train_data[self.features_name]
        )
        # Convert shap values to array , in order to handle shap issues related to
        # native xgboost api
        shap_values = np.array(shap_values)
        dim_shap_values = len(shap_values.shape)
        if self.classif:
            # Check shape_values shape .
            if dim_shap_values == 3:
                #  If it's 3D , then one might extract second axis that contains
                # positive values
                shap_values = shap_values[1]
            else:
                # If it's 2D , nothing to do
                pass
        shap_fig1 = plt.figure()
        shap.summary_plot(
            shap_values, train_data[self.features_name], plot_type="bar", show=False
        )
        shap_fig2 = plt.figure()
        shap.summary_plot(shap_values, train_data[self.features_name], show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data: pd.DataFrame) -> List:
        """
        Computes SHAP force plot for all observations in a giving pandas dataframe
        using Tree Explainer.

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of observations to interpret, must have the same features as the
            model inputs

        Returns
        -------
        List[HTML Object]
        """
        test_data = test_data[self.features_name]
        explainer_model = shap.TreeExplainer(self.model)
        shap_values_model = explainer_model.shap_values(test_data)
        # Convert shap values to array , in order to handle shap issues related to
        # native xgboost api
        shap_values_model = np.array(shap_values_model)
        dim_shap_values = len(shap_values_model.shape)
        # Compute expected value
        expected_value = explainer_model.expected_value
        if self.classif:
            # Check shape_values shape .
            if dim_shap_values == 3:
                #  If it's 3D , then one might extract second axis that contains
                # positive values
                shap_values_model = shap_values_model[1]
                expected_value = expected_value[1]
            else:
                # If it's 2D , nothing to do
                pass
        list_figs = []
        link = "logit" if self.classif else "identity"
        for obs_idx in range(0, len(test_data)):
            logger.info(
                f"Computing SHAP individual plots for {obs_idx + 1}th observation"
            )

            local_fig = shap.force_plot(
                expected_value,
                shap_values_model[obs_idx],
                test_data.iloc[obs_idx],
                link=link,
            )
            list_figs.append(local_fig)
        return list_figs
