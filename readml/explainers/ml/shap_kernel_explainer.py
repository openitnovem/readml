# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import shap

from readml.logger import logger


class ShapKernelExplainer:
    """
    Allows to explain globally or locally any non tree based model using Kernel SHAP
    method.
    Kernel SHAP is a method that uses a special weighted linear regression to compute
    the importance of each feature. The computed importance values are Shapley values
    from game theory and also coefficients from a local linear regression.

    Attributes
    ----------
    model : scikit-learn model or Pyspark model
        Trained model to interpret
    features_name : List[str]
        List of features names used to train the model
    """

    def __init__(self, model: Any, features_name: List[str], classif: bool) -> None:
        self.features_name = features_name
        self.classif = classif
        self.model_func = model.predict_proba if self.classif else model.predict

    def global_explainer(
        self, train_data: pd.DataFrame
    ) -> Tuple[plt.figure, plt.figure]:
        """
        Create a SHAP feature importance plot and SHAP summary plot colored by feature
        values using Kernel Explainer.
        Note that we use shap.kmeans to speed up computations

        Parameters
        ----------
        train_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model
        classif : bool
            True, if it's a classification problem else False

        Returns
        -------
        shap_fig1, shap_fig2 : Tuple[plt.figure, plt.figure]
            SHAP summary plots
        """
        train = train_data[self.features_name]
        train_summary = shap.kmeans(train, 10)
        explainer = shap.KernelExplainer(self.model_func, train_summary)
        shap_values = explainer.shap_values(train)
        if self.classif:
            shap_values = shap_values[1]
        shap_fig1 = plt.figure()
        shap.summary_plot(shap_values, train, show=False)
        shap_fig2 = plt.figure()
        shap.summary_plot(shap_values, train, show=False)
        return shap_fig1, shap_fig2

    def local_explainer(self, test_data: pd.DataFrame):
        """
        Computes SHAP force plot for all observations in a giving pandas dataframe
        using Kernel Explainer.

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of observations to interpret, must have the same features as the
            model inputs
        classif : bool
            True, if it's a classification problem, else False

        Returns
        -------
        HTML Object
        """
        test_data = test_data[self.features_name]
        link = "logit" if self.classif else "identity"
        explainer = shap.KernelExplainer(self.model_func, test_data, link=link)
        shap_values = explainer.shap_values(test_data)

        list_figs = []
        for obs_idx in range(0, len(test_data)):
            logger.info(
                f"Computing SHAP individual plots for {obs_idx + 1}th observation"
            )
            if self.classif:
                local_fig = shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1][obs_idx],
                    test_data.iloc[obs_idx],
                    link="logit",
                )
            else:
                local_fig = shap.force_plot(
                    explainer.expected_value,
                    shap_values[obs_idx],
                    test_data.iloc[obs_idx],
                    link="identity",
                )
            list_figs.append(local_fig)

        return list_figs
