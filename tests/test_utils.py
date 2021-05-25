import csv
import json
import os
from typing import Dict

import pandas as pd
import pytest

from fbd_interpreter.utils import check_and_load_data


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

    os.remove("data_file.csv")
    os.remove("data_file.json")
