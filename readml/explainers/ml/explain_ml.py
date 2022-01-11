import os
from typing import Any, List

import pandas as pd
import shap

from readml.explainers.ml.shap_kernel_explainer import ShapKernelExplainer
from readml.explainers.ml.shap_tree_explainer import ShapTreeExplainer
from readml.icecream import icecream
from readml.logger import ROOT_DIR, logger
from readml.resource.output_builders import initialize_dir
from readml.visualization.plots import interpretation_plots_to_html_report

# Get html sections path
html_sections = os.path.join(ROOT_DIR, "config/sections_html.txt")


class ExplainML:
    """
    Class that contains different interpretability techniques to explain the model
    training (global interpretation) and its predictions (local interpretation).

    Attributes
    ----------
    model : scikit-learn model or Pyspark model
        Model to compute predictions using provided data,
        Any scikit-learn model, `model.predict(data)` must work
        Note that you can also use a pyspark model if you are only using SHAP
        interpretation
    task_name : str
        Task name: choose from supported_tasks in config/config_{type_env}.cfg
    tree_based_model : bool
        If True, we use Tree SHAP algorithms to explain the output of ensemble tree
        models
    features_name : List[str]
        List of features names used to train the model
    features_to_interpret : List[str]
        List of features to interpret using pdp, ice and ale
    target_col : str
        name of target column
    out_path : str
        Output path used to save interpretability plots

    Returns
    -------
    None
    """

    def __init__(
        self,
        model: Any,
        task_name: str,
        tree_based_model: bool,
        features_name: List[str],
        features_to_interpret: List[str],
        target_col: str,
        out_path: str,
    ) -> None:

        self.model = model
        self.features_name = features_name
        self.features_to_interpret = features_to_interpret
        self.target_col = target_col
        self.task_name = task_name
        self.tree_based_model = tree_based_model
        self.out_path = out_path
        self.out_path_global = os.path.join(out_path, "global_interpretation")
        self.out_path_local = os.path.join(out_path, "local_interpretation")
        # Check if output_dir tree exists if not it will be created
        initialize_dir(self.out_path)

    def global_pdp_ice(self, train_data: pd.DataFrame) -> None:
        """
        Compute and save Partial Dependency and Individual Conditional Expectation plots
        using icecream module for global interpretation.

        Parameters
        ----------
        train_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model

        Returns
        -------
        None
        """
        classif = True if self.task_name == "classification" else False
        logger.info("Computing PDP & ice")
        pdp_plots = icecream.IceCream(
            data=train_data.drop([self.target_col], axis=1),
            feature_names=self.features_to_interpret,
            bins=10,
            model=self.model,
            targets=train_data[self.target_col],
            use_classif_proba=classif,
        )
        figs_pdp = pdp_plots.draw(kind="pdp", show=False)
        figs_ice = pdp_plots.draw(kind="ice", show=False)
        logger.info(f"Saving PD plots in {self.out_path_global}")
        interpretation_plots_to_html_report(
            dic_figs=figs_pdp,
            path=os.path.join(self.out_path_global, "partial_dependency_plots.html"),
            title="Partial dependency plots ",
            plot_type="PDP",
            html_sections=html_sections,
        )

        logger.info(f"Saving ICE plots in {self.out_path_global}")
        interpretation_plots_to_html_report(
            dic_figs=figs_ice,
            path=os.path.join(
                self.out_path_global, "individual_conditional_expectation_plots.html"
            ),
            title="Individual Conditional Expectation (ICE) plots ",
            plot_type="ICE",
            html_sections=html_sections,
        )

        return None

    def global_ale(self, train_data: pd.DataFrame) -> None:
        """
        Compute and save Accumulated Local Effect plots using icecream module for global
        interpretation.

        Parameters
        ----------
        train_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model

        Returns
        -------
        None
        """
        classif = True if self.task_name == "classification" else False
        logger.info("Computing ALE")
        ale_plots = icecream.IceCream(
            data=train_data[self.features_name],
            feature_names=self.features_to_interpret,
            bins=10,
            model=self.model,
            targets=train_data[self.target_col],
            use_classif_proba=classif,
            use_ale=True,
        )
        figs_ale = ale_plots.draw(kind="ale", show=False)
        logger.info(f"Saving ALE plots in {self.out_path_global}")
        interpretation_plots_to_html_report(
            dic_figs=figs_ale,
            path=os.path.join(
                self.out_path_global, "accumulated_local_effects_plots.html"
            ),
            title="Accumulated Local Effects (ALE) plots ",
            plot_type="ALE",
            html_sections=html_sections,
        )

        return None

    def global_shap(self, train_data: pd.DataFrame) -> None:
        """
        Compute and save SHAP summary plots for global interpretation.

        Parameters
        ----------
        train_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model

        Returns
        -------
        None
        """
        classif = True if self.task_name == "classification" else False
        logger.info("Computing SHAP")
        if self.tree_based_model:
            logger.info(
                "You are using a tree based model, if it's not the case, "
                "please set tree_based_model to False in "
                "config/config_{type_env}.cfg"
            )

            shap_exp = ShapTreeExplainer(
                model=self.model, features_name=self.features_name, classif=classif
            )
        elif not self.tree_based_model:
            logger.info(
                "You are using a non tree based model, if it's not the case, "
                "please set tree_based_model to True in "
                "config/config_{type_env}.cfg"
            )
            shap_exp = ShapKernelExplainer(
                model=self.model, features_name=self.features_name, classif=classif
            )
        else:
            logger.error(
                "Please set tree_based_model to True or False in "
                "config/config_{type_env}.cfg"
            )
        shap_fig_1, shap_fig_2 = shap_exp.global_explainer(train_data)
        dict_figs = {
            "Summary bar plot": shap_fig_1,
            "Summary bee-swarm plot": shap_fig_2,
        }
        logger.info(f"Saving SHAP plots in {self.out_path_global}")
        interpretation_plots_to_html_report(
            dic_figs=dict_figs,
            path=os.path.join(
                self.out_path_global, "shap_feature_importance_plots.html"
            ),
            title="SHAP feature importance plots",
            plot_type="SHAP",
            html_sections=html_sections,
        )

        return None

    def local_shap(self, test_data: pd.DataFrame) -> None:
        """
        Compute and save SHAP force plots for all observations in a test_data for local
        interpretation.

        Parameters
        ----------
        test_data : pd.DataFrame
            Dataframe of model inputs, used to explain the model predictions

        Returns
        -------
        None
        """
        classif = True if self.task_name == "classification" else False

        if self.tree_based_model:
            logger.info(
                "You are using a tree based model, if it's not the case, please set "
                "tree_based_model to False in "
                "config/config_{type_env}.cfg"
            )

            shap_exp = ShapTreeExplainer(
                model=self.model, features_name=self.features_name, classif=classif
            )

        elif not self.tree_based_model:
            logger.info(
                "You are using a non tree based model, if it's not the case, please set"
                " tree_based_model to True in "
                "config/config_{type_env}.cfg"
            )
            shap_exp = ShapKernelExplainer(
                model=self.model, features_name=self.features_name, classif=classif
            )

        else:
            logger.error(
                "Please set tree_based_model to True or False in "
                "config/config_{type_env}.cfg"
            )
        list_figs = shap_exp.local_explainer(test_data=test_data)

        for j in range(0, len(test_data)):
            logger.info(
                f"Saving SHAP individual plots for {j + 1}th observation in "
                "{self.out_path_local}"
            )
            shap.save_html(
                os.path.join(
                    self.out_path_local,
                    f"shap_local_explanation_{j + 1}th_obs.html",
                ),
                list_figs[j],
            )

        return None
