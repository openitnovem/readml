# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

import pandas as pd

from readml import config
from readml.config import env
from readml.config.load import load_cfg_resource
from readml.logger import logger
from readml.resource.data_loader import (
    load_csv_resource,
    load_json_resource,
    load_parquet_resource,
)


def _parse_and_check_config() -> Dict[str, str]:
    """Parse config from cfg file and return dictionnary with keys as config_params

    Returns
    -------
    dico_params : Dict[str, str]
        configuration as dictionnary

    Example
    -------
    >>> conf = _parse_and_check_config()
    >>> len(list(conf.keys())) > 0
    True
    """
    # Get configuration as dict from config_{type_env}.cfg
    config_ = load_cfg_resource(config, f"config_{env}.cfg")
    configuration: dict = {s: dict(config_.items(s)) for s in config_.sections()}
    dico_params = {}
    # Get model path as pickle (if ML) or h5 (if DL)
    dico_params["model_path"] = configuration["PARAMS"]["model_path"]
    # Get output path as str
    dico_params["out_path"] = configuration["PARAMS"]["output_path"]
    # Get task name as str
    dico_params["task_name"] = configuration["PARAMS"]["task_name"]
    # Get learning type
    dico_params["learning_type"] = configuration["PARAMS"]["learning_type"]
    # Get data type as str (image, text or tabular)
    dico_params["data_type"] = configuration["PARAMS"]["data_type"]
    learning_type = dico_params["learning_type"]
    data_type = dico_params["data_type"]
    if learning_type == "ML" and data_type == "tabular":
        dico_params = _parse_conf_ml_tab(dico_params, configuration)
    elif learning_type == "DL" and data_type == "tabular":
        dico_params = _parse_conf_dl_tab(dico_params, configuration)
    elif learning_type == "DL" and data_type == "text":
        dico_params = _parse_conf_dl_text(dico_params, configuration)
    elif learning_type == "DL" and data_type == "image":
        dico_params = _parse_conf_dl_image(dico_params, configuration)

    # Sanity check
    mandatory_conf = [
        "learning_type",
        "data_type",
        "model_path",
        "task_name",
    ]
    mandatory_ml = ["features_to_interpret", "tree_based_model"]
    mandatory_tabular = [
        "features_name",
        "target_col",
    ]
    mandatory_text = ["word2index_path"]
    mandatory_image = ["images_folder_path"]
    if learning_type == "ML":
        mandatory_conf = mandatory_conf + mandatory_ml
    if data_type == "tabular":
        mandatory_conf = mandatory_conf + mandatory_tabular
    elif data_type == "text":
        mandatory_conf = mandatory_conf + mandatory_text
    elif data_type == "image":
        mandatory_conf = mandatory_conf + mandatory_image
    missing_conf = False

    for k in mandatory_conf:
        if dico_params[k] == "":
            logger.error(f"Configuration  requires {k} , but is missing ")
            missing_conf = True
    if missing_conf:
        raise KeyError(
            "Missing configuration , please update conf file located in "
            "config/config_{type_env}.cfg by filling in missing keys "
        )
    return dico_params


def _parse_conf_ml_tab(dico_params: Dict[str, str], configuration) -> Dict[str, str]:
    dico_params = _parse_conf_dl_tab(dico_params)
    # Get features to interpret as list
    dico_params["features_to_interpret"] = configuration["PARAMS"][
        "features_to_interpret"
    ]
    # Get model type (tree based model or not)
    dico_params["tree_based_model"] = configuration["PARAMS"]["tree_based_model"]
    return dico_params


def _parse_conf_dl_tab(dico_params: Dict[str, str], configuration) -> Dict[str, str]:
    dico_params = _parse_tab_and_text(dico_params)
    # Get features name as list
    dico_params["features_name"] = configuration["PARAMS"]["features_name"]
    return dico_params


def _parse_conf_dl_text(dico_params: Dict[str, str], configuration) -> Dict[str, str]:
    dico_params = _parse_tab_and_text(dico_params)
    # Get word2index path name as list
    dico_params["word2index_path"] = configuration["PARAMS"]["word2index_path"]
    return dico_params


def _parse_tab_and_text(dico_params: Dict[str, str], configuration) -> Dict[str, str]:
    # Get train data path as csv / parquet
    dico_params["train_data_path"] = configuration["PARAMS"]["train_data_path"]
    # Get train data format to load the data
    dico_params["train_data_format"] = configuration["PARAMS"]["train_data_format"]
    # Get test data path as csv / parquet
    dico_params["test_data_path"] = configuration["PARAMS"]["test_data_path"]
    # Get test data format to load the data
    dico_params["test_data_format"] = configuration["PARAMS"]["test_data_format"]
    # Get target column name as str
    dico_params["target_col"] = configuration["PARAMS"]["target_col"]
    return dico_params


def _parse_conf_dl_image(dico_params: Dict[str, str], configuration) -> Dict[str, str]:
    # Get images folder path as str
    dico_params["images_folder_path"] = configuration["PARAMS"]["images_folder_path"]
    # Get image size and color mode
    dico_params["img_height"] = configuration["PARAMS"]["img_height"]
    dico_params["img_width"] = configuration["PARAMS"]["img_width"]
    dico_params["color_mode"] = configuration["PARAMS"]["color_mode"]
    return dico_params


def check_and_load_data(data_path: str, data_format: str, data_type: str):
    if data_path == "" or data_format == "":
        logger.error(
            f"Configuration file requires {data_type} data path and format, "
            "but is missing "
        )
        raise KeyError(
            f"Missing {data_type} data path or format, please update conf file located "
            "in config/config_[type_env].cfg by filling {data_type}_data_path "
        )
    elif data_format == "parquet":
        data = load_parquet_resource(data_path)
    elif data_format == "csv":
        data = load_csv_resource(data_path)
    else:
        data = load_json_resource(data_path)
    return data


def optimize(
    data: pd.DataFrame,
    datetime_features: List[str] = [],
    datetime_format: str = "%Y%m%d",
    prop_unique: float = 0.5,
):
    """
    Returns a pandas dataframe with better memory allocation by downcasting the columns
    automatically to the smallest possible datatype without losing any information.
    For strings we make use of the pandas category column type if the amount of unique
    strings cover less than half the total amount of strings.
    We cast date columns to the pandas datetime dtype. It does not reduce memory usage,
    but enables time based operations.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe to reduce
    datetime_features : List[str]
        List of date columns to cast to the pandas datetime dtype
    datetime_format : str, optional
        datetime features format (the default is "%Y%m%d")
    prop_unique : float, optional = 0.5
        max proportion of unique values in object columns to allow casting to category
        type (the default is 0.5)

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> data = pd.DataFrame(d)
    >>> print(optimize(data).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int8', 'col2': 'float32', 'col3': 'category'}
    """
    optimized_df = optimize_floats(
        optimize_ints(
            optimize_objects(data, datetime_features, datetime_format, prop_unique)
        )
    )
    return optimized_df


def optimize_floats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the float columns to the smallest
     possible float datatype (float32, float64) using pd.to_numeric.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe to reduce

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for float columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> data = pd.DataFrame(d)
    >>> print(optimize_floats(data).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int64', 'col2': 'float32', 'col3': 'object'}
    """
    floats = data.select_dtypes(include=["float64"]).columns.tolist()
    data[floats] = data[floats].apply(pd.to_numeric, downcast="float")
    return data


def optimize_ints(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the integer columns to the smallest
     possible int datatype (int8, int16, int32, int64) using pd.to_numeric.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe to reduce

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for int columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> data = pd.DataFrame(d)
    >>> print(optimize_ints(data).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int8', 'col2': 'float64', 'col3': 'object'}
    """
    ints = data.select_dtypes(include=["int64"]).columns.tolist()
    data[ints] = data[ints].apply(pd.to_numeric, downcast="integer")
    return data


def optimize_objects(
    data: pd.DataFrame,
    datetime_features: List[str],
    datetime_format: str = "%Y%m%d",
    prop_unique: float = 0.5,
) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the object columns to the smallest
    possible datatype.
    For strings we make use of the pandas category column type if the amount of unique
    strings cover less than the proportion p (default 50%) of the total amount of
    strings.
    We cast date columns to the pandas datetime dtype. It does not reduce memory usage,
    but enables time based operations.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas dataframe to reduce
    datetime_features : List[str]
        List of date columns to cast to the pandas datetime dtype
    datetime_format : str, optional
        datetime features format (the default is "%Y%m%d")
    prop_unique : float, optional
        max proportion of unique values to allow casting to category type
        (the default is 0.5)

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for object columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': ["08:10", "10:15", "12:30", "06:00"], 'col2': ["M", "F", "M", "F"]}
    >>> data = pd.DataFrame(d)
    >>> print(optimize_objects(data, 'col1', "%H:%M").dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'datetime64[ns]', 'col2': 'category'}
    """

    for col in data.select_dtypes(include=["object"]):
        if col not in datetime_features:
            num_unique_values = len(data[col].unique())
            num_total_values = len(data[col])
            if float(num_unique_values) / num_total_values <= prop_unique:
                data[col] = data[col].astype("category")
        else:
            data[col] = pd.to_datetime(data[col], format=datetime_format)
    return data
