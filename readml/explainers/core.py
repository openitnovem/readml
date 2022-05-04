from typing import Dict


from readml.explainers.dl.explain_dl import ExplainDL
from readml.explainers.ml.explain_ml import ExplainML
from readml.logger import logger
from readml.resource.data_loader import load_json_resource, load_pickle_resource
from readml.utils import check_and_load_data, optimize

import tensorflow as tf

def interpret_ml(
    config_values: Dict[str, str],
    interpret_type: str = "mix",
    use_ale: bool = True,
    use_pdp_ice: bool = True,
    use_shap: bool = True,
) -> None:
    """
    Interpret locally, globally or both any ML model using PDP, ICE, ALE & SHAP.
    Note that to speed up computations, we apply the function `optimize` to reduce the pandas
    dataframes memory usage, by downcasting the columns automatically to the smallest possible
    datatype without losing any information.
    Outputs are saved in the given output path from the config file.

    Parameters
    ----------
    config_values: Dict[str, str]
        configuration as dictionnary from the config file located in
        config/config_{type_env}.cfg
    interpret_type : str, optional
        Type of interpretability global, local or mix(both). (the default is "mix", which implies
        global and local interpretability)
    use_ale : bool, optional
        If True, computes ALE: Accumulated Local Effects.
        Can only be used for numerical features. (the default is True)
    use_pdp_ice : bool, optional
        If True, computes PDP & ICE: Partial Dependency & Individual Expectation plots.
        (the default is True)
    use_shap : bool, optional
        If True, computes SHAP plots. (the default is True)

    Returns
    -------
    None
    """
    logger.info("Loading ML model")
    model = load_pickle_resource(config_values["model_path"])
    tree_based_model = True if config_values["tree_based_model"] == "True" else False

    exp = ExplainML(
        model=model,
        task_name=config_values["task_name"],
        tree_based_model=tree_based_model,
        features_name=config_values["features_name"].split(","),
        features_to_interpret=config_values["features_to_interpret"].split(","),
        target_col=config_values["target_col"],
        out_path=config_values["out_path"],
    )
    if interpret_type == "global" or interpret_type == "mix":
        logger.info("Interpretability type : global")
        train_data_path = config_values["train_data_path"]
        train_data_format = config_values["train_data_format"]
        logger.info("Loading train data")
        train_data = check_and_load_data(
            data_path=train_data_path, data_format=train_data_format, data_type="train"
        )
        logger.info("Reducing train dataframe memory usage to speed up computations")
        train_data = optimize(train_data)
        if use_pdp_ice:
            exp.global_pdp_ice(train_data)
        if use_ale:
            exp.global_ale(train_data)
        if use_shap:
            exp.global_shap(train_data)

    if interpret_type == "local" or interpret_type == "mix":
        logger.info("Interpretability type : local")
        test_data_path = config_values["test_data_path"]
        test_data_format = config_values["test_data_format"]
        logger.info("Loading test data")
        test_data = check_and_load_data(
            data_path=test_data_path, data_format=test_data_format, data_type="test"
        )
        logger.info("Reducing test dataframe memory usage to speed up computations")
        test_data = optimize(test_data)
        exp.local_shap(test_data)

    else:
        raise Exception  # Not supported


def interpret_dl(config_values: Dict[str, str]) -> None:
    """
    Interpret locally any DL model based on user configuration from the file located in
    config/config_{type_env}.cfg
    Outputs are saved in the given output path from the config file.

    Parameters
    ----------
    config_values: Dict[str, str]
        configuration as dictionary from the config file located in
        config/config_{type_env}.cfg

    Returns
    -------
    None
    """

    data_type = config_values["data_type"]
    if data_type != "image":
        tf.compat.v1.disable_v2_behavior()

    logger.info("Loading DL model")
    model = tf.keras.models.load_model(config_values["model_path"], compile=False)
    exp = ExplainDL(model=model, out_path=config_values["out_path"])
    logger.info(f"Data type : {data_type}")
    if data_type == "image":
        exp.explain_image(
            image_dir=config_values["images_folder_path"],
            size=(int(config_values["img_height"]), int(config_values["img_width"])),
            color_mode=config_values["color_mode"],
        )
    else:
        test_data_path = config_values["test_data_path"]
        test_data_format = config_values["test_data_format"]
        logger.info("Loading test data")
        test_data = check_and_load_data(
            data_path=test_data_path, data_format=test_data_format, data_type="test"
        )
        if data_type == "tabular":
            exp.explain_tabular(
                test_data=test_data,
                features_name=config_values["features_name"].split(","),
                task_name=config_values["task_name"],
            )

        elif data_type == "text":
            # Load word2idx from json file
            logger.info("Loading word to index json file")
            word2idx = load_json_resource(
                resource_file_name=config_values["word2index_path"]
            )
            exp.explain_text(
                test_data=test_data.head(10),
                target_col=config_values["target_col"],
                word2idx=word2idx,
            )
