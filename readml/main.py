import json
from pprint import pformat
from typing import Dict, Optional

import click

from readml.explainers.core import interpret_dl, interpret_ml
from readml.logger import logger
from readml.utils import _parse_and_check_config


@click.command()
@click.option(
    "--config-values",
    default=None,
    show_default=True,
    metavar="",
    help="Dictionnary of configuration adapted to the type of problematic you need to interpret",
)
@click.option(
    "--interpret-type",
    default="mix",
    show_default=True,
    metavar="",
    help="Interpretability type: Choose global, local or mix. Not needed for DL",
)
@click.option(
    "--use-ale",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots ALE. Not needed for DL",
)
@click.option(
    "--use-pdp-ice",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots PDP & ICE. Not needed for DL",
)
@click.option(
    "--use-shap",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots shapely values for global & local explanation. Not needed for DL",
)
def interpret(
    config_values: Optional[Dict[str, str]] = None,
    interpret_type: str = "mix",
    use_ale: bool = True,
    use_pdp_ice: bool = True,
    use_shap: bool = True,
) -> None:
    """
    Interpret any model (ML & DL).
    Before using this function, you must fill in the config file located in
    config/config_{type_env}.cfg.
    Outputs are saved in the given output path from the config file.

    Parameters
    ----------
    config_values: Dict[str, str], optional
        Dictionnary values which needs to contain keys and values corresponding to the type of problematic.
        You can use the template in ./readml/config/ as examples to build your dictionnaries.
    interpret_type : str, optional
        Type of interpretability global, local or mix(both). (the default is "mix", which implies
        global and local interpretability)
        Not needed for DL models.
    use_ale : bool, optional
        If True, computes ALE: Accumulated Local Effects.
        Can only be used for numerical features. (the default is True)
        Not needed for DL models.
    use_pdp_ice : bool, optional
        If True, computes PDP & ICE: Partial Dependency & Individual Expectation plots.
        (the default is True)
        Not needed for DL models.
    use_shap : bool, optional
        If True, computes SHAP plots. (the default is True)
        Not needed for DL models.

    Returns
    -------
    None
    """
    if config_values is None:
        config_values = _parse_and_check_config()
    else:
        config_values = json.loads(config_values)
    logger.info(f"Configuration settings :\n{pformat(config_values)}")
    learning_type = config_values["learning_type"]
    logger.info(f"Learning type is {learning_type}")
    if learning_type == "ML":
        interpret_ml(config_values, interpret_type, use_ale, use_pdp_ice, use_shap)
    elif learning_type == "DL":
        interpret_dl(config_values)
    else:
        logger.error(
            "Configuration file requires a valid learning_type, but is missing"
        )
        raise KeyError(
            "Invalid learning_type, please update conf file located in config/config_[type_env].cfg"
            " by filling learning_type wit ML or DL "
        )


if __name__ == "__main__":
    # After the use of pytest, you can run :
    # For tabular ML : python3 readml/main.py --config-values='{"model_path": "/workspaces/readml/outputs/tests/core/model/model.sav" ,"out_path": "/workspaces/readml/outputs/tests/core/","task_name": "regression","learning_type": "ML","data_type": "tabular","features_to_interpret": "F1,F2","tree_based_model": "True","features_name": "F1,F2,F3","target_col": "target","train_data_path": "/workspaces/readml/outputs/tests/core/data/train.csv","train_data_format": "csv","test_data_path": "/workspaces/readml/outputs/tests/core/data/train.csv","test_data_format": "csv"}'
    # For tabular DL image : python3 readml/main.py --config-values='{"model_path": "/workspaces/readml/outputs/tests/dl/model/","out_path": "/workspaces/readml/outputs/tests/core/","task_name": "regression","learning_type": "DL","data_type": "image","images_folder_path": "/workspaces/readml/outputs/tests/dl/data_image/","img_height": "32","img_width": "32","color_mode": "rgb"}'
    interpret()
