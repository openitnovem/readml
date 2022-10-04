# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List

import pandas as pd
import shap

from readml.logger import logger


class TabularExplainer:
    """
    Allows to explain a DL model predictions on tabular data using Deep SHAP.

    Attributes
    ----------
    model : TensorFlow model
        TensorFlow models and Keras models using the TensorFlow backend are supported
        Model to compute predictions using provided data.
    features_name : List[str]
        List of features names used to train the model
    classif : bool
        True if classification task, False if regression
    """

    def __init__(self, model: Any, features_name: List[str], classif: bool) -> None:
        self.model = model
        self.features_name = features_name
        self.link = "logit" if classif else "identity"

    def local_explainer(self, test_data: pd.DataFrame) -> List:
        """
        Computes SHAP force plots for all observations of a given pandas dataframe using
        Deep Explainer.

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of observations to interpret, must have the same features as the
            model inputs

        Returns
        -------
        list_figs : List[HTML Object]
            List of shap force plots for all test_data observations
        """
        test_data = test_data[self.features_name]
        x_test = test_data.to_numpy(dtype="float", na_value=0)
        explainer = shap.DeepExplainer(model=self.model, data=x_test)
        shap_values = explainer.shap_values(x_test)
        list_figs = []
        for obs_idx in range(0, len(test_data)):
            logger.info(
                f"Computing SHAP individual plots for {obs_idx + 1}th observation"
            )
            local_fig = shap.force_plot(
                explainer.expected_value,
                shap_values[0][obs_idx],
                test_data.iloc[obs_idx],
                link=self.link,
            )
            list_figs.append(local_fig)
        return list_figs
