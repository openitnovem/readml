import csv
import io
import json
import os
from typing import Dict

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from readml.config import get_interp_env
from readml.utils import _parse_and_check_config, check_and_load_data, optimize

DICT_DTYPES = {
    "MedInc": "float32",
    "HouseAge": "float32",
    "AveRooms": "float32",
    "AveBedrms": "float32",
    "Population": "float32",
    "AveOccup": "float32",
    "Latitude": "float32",
    "Longitude": "float32",
    "Text": "object",
    "Integer": "int8",
}


def create_data_to_optimize():
    dict_data = fetch_california_housing()
    data = pd.DataFrame(dict_data["data"], columns=dict_data["feature_names"]).query(
        "index < 10"
    )
    data["Text"] = ["test" + str(iterate) for iterate in range(data.shape[0])]
    data["Integer"] = [iterate for iterate in range(data.shape[0])]
    return data


def extract_dtype(data_dtype):
    res = data_dtype.to_dict()
    new_res = {}
    for key, value in res.items():
        new_res[key] = str(value)
    return new_res


def test_optimize():
    data = create_data_to_optimize()
    dtype_brut = extract_dtype(data.dtypes)
    data = optimize(data)
    dtype_optimize = extract_dtype(data.dtypes)
    assert dtype_brut != dtype_optimize
    assert dtype_optimize == DICT_DTYPES


def test_check_and_load_data():
    with pytest.raises(Exception) as e_info:
        _ = check_and_load_data(data_path="", data_format="", data_type="empty")
    assert isinstance(e_info.value, KeyError)

    with open("data_file.csv", "w") as csvfile:
        data_csv = csv.writer(csvfile)
        data_csv.writerow(["Test"] * 5)
    data_csv = check_and_load_data(
        data_path="data_file.csv", data_format="csv", data_type="csv file"
    )
    assert isinstance(data_csv, pd.DataFrame)

    dict = {"a": 1, "b": 2}
    with open("data_file.json", "w") as jsonfile:
        json.dump(dict, jsonfile)
    data_json = check_and_load_data(
        data_path="data_file.json", data_format="json", data_type="json file"
    )
    assert isinstance(data_json, Dict)

    data = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    data.to_parquet("data_file.parquet")
    data_parquet = check_and_load_data(
        data_path="data_file.parquet", data_format="parquet", data_type="parquet file"
    )
    assert isinstance(data_parquet, pd.DataFrame)

    parquet_file = io.BytesIO()
    data.to_parquet(parquet_file)
    parquet_file.seek(0)
    assert isinstance(parquet_file.read(), bytes)

    os.remove("data_file.csv")
    os.remove("data_file.json")
    os.remove("data_file.parquet")


CONFIG_DL_IMAGE = {
    "model_path": "",
    "out_path": "",
    "task_name": "classification",
    "learning_type": "DL",
    "data_type": "image",
    "images_folder_path": "",
    "img_height": 224,
    "img_width": 224,
    "color_mode": "rgb",
}


def test_parse_and_check_config():
    extract_config = _parse_and_check_config()
    assert extract_config["task_name"] in ["regression", "classification"]
    assert extract_config["learning_type"] in ["ML", "DL"]
    assert extract_config["data_type"] in ["tabular", "text", "image"]
    if get_interp_env() == "local":
        assert list(extract_config.keys()) == list(CONFIG_DL_IMAGE.keys())
