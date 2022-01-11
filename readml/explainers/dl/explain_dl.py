import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import shap

from readml.explainers.dl.image_deep_explainer import VisualExplainer
from readml.explainers.dl.tabular_deep_explainer import TabularExplainer
from readml.explainers.dl.text_deep_explainer import TextExplainer
from readml.logger import logger


class ExplainDL:
    """
    Class that contains different interpretability techniques to explain the DL model predictions
    depending on its data type and saves interpretability plots in out_path.

    Attributes
    ----------
    model : TensorFlow model
        TensorFlow models and Keras models using the TensorFlow backend are supported
        Model to compute predictions using provided data.
    out_path : str
        Output path used to save interpretability plots.
    """

    def __init__(self, model: Any, out_path: str) -> None:
        self.model = model
        self.out_path = out_path

    def explain_tabular(
        self, test_data: pd.DataFrame, features_name: List[str], task_name: str
    ) -> None:
        """
        Explain the predictions of a DL model on tabular data using SHAP DeepExplainer (Deep SHAP).

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of observations to explain (must contain all features used to train the model)
        features_name : List[str]
            List of features names used to train the model
        task_name : {"classification", "regression"}
            Task name: choose from supported_tasks in config/config_{type_env}.cfg

        Returns
        -------
        None
        """
        classif = True if task_name == "classification" else False

        tab_explain = TabularExplainer(
            model=self.model, features_name=features_name, classif=classif
        )
        list_figs = tab_explain.local_explainer(test_data=test_data)

        for j in range(0, len(test_data)):
            logger.info(
                f"Saving SHAP individual plots for {j + 1}th observation in {self.out_path}"
            )
            shap.save_html(
                os.path.join(
                    self.out_path, f"tab_deep_local_explanation_{j + 1}th_obs.html"
                ),
                list_figs[j],
            )

    def explain_text(
        self, test_data: pd.DataFrame, target_col: str, word2idx: Dict[str, int]
    ) -> None:
        """
        Explain the predictions of a DL model on text data using SHAP DeepExplainer and
        a word2index mapping.

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of observations to explain (must contain all features used to train the model
        target_col : str
            name of target column, used to drop target column and get only features
        word2idx : Dict[str, int]
            Dict where keys are the vocabulary and values the index.
        Returns
        -------
        None
        """
        text_explain = TextExplainer(
            model=self.model,
            target=target_col,
            word2idx=word2idx,
        )
        list_figs = text_explain.local_explainer(test_data=test_data)
        for j in range(0, len(test_data)):
            logger.info(
                f"Saving SHAP individual plots for {j + 1}th observation in {self.out_path}/"
            )

            shap.save_html(
                os.path.join(
                    self.out_path, f"text_deep_local_explanation_{j + 1}th_obs.html"
                ),
                list_figs[j],
            )

    def explain_image(
        self, image_dir: str, size: Tuple[int, int], color_mode: str
    ) -> None:
        """
        Explain the predictions of a DL model on image data using GRAD-CAM.

        Parameters
        ----------
        image_dir : str
            Path of the directory containing images to explain.
        size : Tuple[int,int]
            Tuple of ints (img_height, img_width)
        color_mode : {"grayscale","rgb"}
            color mode, choose "grayscale" or "rgb"

        Returns
        -------
        None
        """
        image_explain = VisualExplainer(model=self.model)
        image_explain.local_explainer(
            img_dir=os.path.abspath(image_dir),
            path_out=self.out_path,
            size=size,
            color_mode=color_mode,
        )
