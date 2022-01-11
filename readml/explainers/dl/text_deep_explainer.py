from __future__ import print_function

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from readml.logger import logger


class TextExplainer:
    """
    Allows to explain a DL model predictions on text data using shap deep explainer and
    a word to index mapping.

    Attributes
    ----------
    model : TensorFlow model
        TensorFlow models and Keras models using the TensorFlow backend are supported
        Model to compute predictions using provided data.
    target : str
        Target column name. If test_data doesn't have a target column, set target to ""
    word2idx : Dict[str,int]
        Dict where keys are the vocabulary and values the index.
    """

    def __init__(self, model: Any, target: str, word2idx: Dict[str, int]) -> None:
        self.model = model
        self.target = target
        self.word2idx = word2idx

    def local_explainer(self, test_data: pd.DataFrame) -> List:
        """
        Computes SHAP force plots for all observations of a given pandas dataframe using
        Deep Explainer.
        Uses the word to index mapping to get features names

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
        if self.target != "":
            test_data = test_data.drop(self.target, axis=1)
        x_test = test_data.to_numpy()

        explainer = shap.DeepExplainer(self.model, x_test)
        shap_values = explainer.shap_values(x_test)
        words = self.word2idx
        num2word = {}
        for word in words.keys():
            num2word[words[word]] = word
        x_test_words = np.stack(
            [
                np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i])))
                for i in range(len(x_test))
            ]
        )
        list_figs = []
        for obs_idx in range(0, len(x_test)):
            logger.info(
                f"Computing SHAP individual plots for {obs_idx + 1}th observation"
            )
            local_fig = shap.force_plot(
                explainer.expected_value,
                shap_values[0][obs_idx],
                x_test_words[obs_idx],
                link="logit",
            )
            list_figs.append(local_fig)
        return list_figs
