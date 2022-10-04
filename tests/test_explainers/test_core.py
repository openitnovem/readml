# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from readml.explainers.core import interpret_ml
from readml.logger import ROOT_DIR

PREDICTIONS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
TARGETS = pd.Series(PREDICTIONS)
DATA = pd.DataFrame(
    {
        "F1": [0, 1, 2, 3, 3, 4, 5, 6, 7, 7, 0, 1, 2, 3, 3, 4, 5, 6, 7, 7],
        "F2": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "F3": [0, 1, 1, 0, 1, 2, 3, 2, 5, 2, 0, 1, 1, 0, 1, 2, 3, 2, 5, 2],
        "target": PREDICTIONS,
    }
)


def initialize_directories(out_path, dir_to_create):
    os.chdir(ROOT_DIR)
    new_root = os.getcwd()
    new_root = "/".join(new_root.split("/")[:-1])
    os.chdir(new_root)
    start = out_path.index("/") + 1
    split = out_path[start:].split("/")[:-1]
    for elt in split:
        if not os.path.isdir(elt):
            os.makedirs(elt)
            os.chdir(elt)
        else:
            os.chdir(elt)
    os.chdir(os.path.join(ROOT_DIR, out_path))
    for elt in dir_to_create:
        if not os.path.isdir(elt):
            os.makedirs(elt)


dir_to_create = ["data", "model", "intelligibility"]
out_path = "../outputs/tests/core/"
initialize_directories(out_path, dir_to_create)
data_train_path = os.path.join(ROOT_DIR, "../outputs/tests/core/", "data/train.csv")
model_path = os.path.join(ROOT_DIR, "../outputs/tests/core/", "model/model.sav")
output_path = os.path.join(ROOT_DIR, "../outputs/tests/core/", "intelligibility")


CONFIG_ML = {
    "model_path": model_path,
    "out_path": output_path,
    "task_name": "regression",
    "learning_type": "ML",
    "data_type": "tabular",
    "features_to_interpret": "F1,F2",
    "tree_based_model": "True",
    "features_name": "F1,F2,F3",
    "target_col": "target",
    "train_data_path": data_train_path,
    "train_data_format": "csv",
    "test_data_path": data_train_path,
    "test_data_format": "csv",
}


def generate_use_case_ml():
    DATA.to_csv(data_train_path, index=False)
    model = RandomForestRegressor(n_estimators=10)
    model.fit(DATA.filter(items=["F1", "F2", "F3"]), DATA["target"])
    pickle.dump(model, open(model_path, "wb"))


def test_interpret_ml():
    generate_use_case_ml()

    global_path = os.path.join(output_path, "global_interpretation")
    local_path = os.path.join(output_path, "local_interpretation")
    if os.path.exists(global_path):
        for files in os.listdir(global_path):
            os.remove(os.path.join(global_path, files))
    if os.path.exists(local_path):
        for files in os.listdir(local_path):
            os.remove(os.path.join(local_path, files))

    interpret_ml(CONFIG_ML)

    output_path_pdp = os.path.join(global_path, "partial_dependency_plots.html")
    output_path_ice = os.path.join(
        global_path,
        "individual_conditional_expectation_plots.html",
    )
    output_path_global_shap = os.path.join(
        global_path, "shap_feature_importance_plots.html"
    )
    output_path_ale = os.path.join(global_path, "accumulated_local_effects_plots.html")

    min_obs, max_obs = 1, DATA.shape[0]
    output_path_local_min_obs_shap = os.path.join(
        local_path,
        f"shap_local_explanation_{min_obs}th_obs.html",
    )
    output_path_local_max_obs_shap = os.path.join(
        local_path,
        f"shap_local_explanation_{max_obs}th_obs.html",
    )
    outside_output = os.path.join(
        local_path,
        f"shap_local_explanation_{max_obs + 1}th_obs.html",
    )

    assert os.path.isfile(output_path_local_min_obs_shap)
    assert os.path.isfile(output_path_local_max_obs_shap)
    assert not os.path.isfile(outside_output)
    assert os.path.isfile(output_path_ale)
    assert os.path.isfile(output_path_pdp)
    assert os.path.isfile(output_path_ice)
    assert os.path.isfile(output_path_global_shap)
