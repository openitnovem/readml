from pprint import pformat

import click

from fbd_interpreter.explainers.core import interpret_dl, interpret_ml
from fbd_interpreter.logger import logger
from fbd_interpreter.utils import _parse_and_check_config


@click.command()
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
    config_values = _parse_and_check_config()
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
    interpret()
