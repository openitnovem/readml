# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
import sklearn

from readml.explainers.ml.explain_ml import ExplainML
from readml.logger import ROOT_DIR

FEATURES = ["a", "b"]
PREDICTIONS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
TARGETS_Y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
DATA = pd.DataFrame(
    {
        "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "b": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "target": TARGETS_Y,
    }
)
DATA_X = DATA[FEATURES]
PREDICT_PROBA = np.array(
    [
        [
            0.6,
            0.8,
            0.9,
            0.7,
            0.6,
            0.8,
            0.9,
            0.7,
            0.8,
            0.1,
            0.2,
            0.4,
            0.3,
            0.1,
            0.2,
            0.4,
            0.3,
            0.2,
        ],
        [
            0.4,
            0.2,
            0.1,
            0.3,
            0.4,
            0.2,
            0.1,
            0.3,
            0.2,
            0.9,
            0.8,
            0.6,
            0.7,
            0.9,
            0.8,
            0.6,
            0.7,
            0.8,
        ],
    ]
).T


def initialize_directories(out_path, dir_to_create):
    os.chdir(ROOT_DIR)
    new_root = os.getcwd()
    new_root = "/".join(new_root.split("/")[:-1])
    os.chdir(new_root)
    start = out_path.index("/") + 1
    split = out_path[start:].split("/")
    for elt in split:
        if not os.path.isdir(elt):
            os.makedirs(elt)
            os.chdir(elt)
        else:
            os.chdir(elt)
    os.chdir(new_root)

    for elt in dir_to_create:
        if not os.path.isdir(os.path.join(out_path, elt)):
            os.makedirs(os.path.join(out_path, elt))


dir_to_create = ["local_interpretation", "global_interpretation"]
out_path = "outputs/tests/ml"
initialize_directories(out_path, dir_to_create)
output_path_local_dir = os.path.join(
    ROOT_DIR, "../outputs/tests/ml", "local_interpretation"
)
output_path_global_dir = os.path.join(
    ROOT_DIR, "../outputs/tests/ml", "global_interpretation"
)


class DummyModel(object):
    def __init__(self, classification: bool = True):
        self.predict = lambda x: PREDICTIONS
        if classification:
            self.classes_ = [0, 1]
            self.predict_proba = lambda x: PREDICT_PROBA


interpreter = ExplainML(
    model=DummyModel(),
    task_name="classification",
    tree_based_model=False,
    features_name=FEATURES,
    features_to_interpret=FEATURES,
    target_col="target",
    out_path=os.path.join(ROOT_DIR, "../outputs/tests/ml"),
)


def test_global_pdp_ice() -> None:
    interpreter.global_pdp_ice(DATA)
    output_path_pdp = os.path.join(
        output_path_global_dir, "partial_dependency_plots.html"
    )
    output_path_ice = os.path.join(
        output_path_global_dir, "individual_conditional_expectation_plots.html"
    )
    assert os.path.isfile(output_path_pdp)
    assert os.path.isfile(output_path_ice)


def test_global_ale() -> None:
    interpreter.global_ale(DATA)
    output_path_ale = os.path.join(
        output_path_global_dir, "accumulated_local_effects_plots.html"
    )
    assert os.path.isfile(output_path_ale)


def kernel_interp(scale: str) -> None:
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(DATA_X, TARGETS_Y)
    kernel_interpreter = ExplainML(
        model=svm,
        task_name="classification",
        tree_based_model=False,
        features_name=DATA_X.columns.tolist(),
        features_to_interpret=[],
        target_col="",
        out_path=os.path.join(ROOT_DIR, "../outputs/tests/ml"),
    )
    if scale == "global":
        kernel_interpreter.global_shap(DATA_X)
    elif scale == "local":
        kernel_interpreter.local_shap(DATA_X)


def tree_interp(scale: str) -> None:
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    model.fit(DATA_X, TARGETS_Y)

    tree_interpreter = ExplainML(
        model=model,
        task_name="classification",
        tree_based_model=True,
        features_name=DATA_X.columns.tolist(),
        features_to_interpret=[],
        target_col="",
        out_path=os.path.join(ROOT_DIR, "../outputs/tests/ml"),
    )
    if scale == "global":
        tree_interpreter.global_shap(DATA_X)
    elif scale == "local":
        tree_interpreter.local_shap(DATA_X)


def test_global_shap() -> None:
    kernel_interp("global")
    output_path_global_kernel_shap = os.path.join(
        output_path_global_dir, "shap_feature_importance_plots.html"
    )
    assert os.path.isfile(output_path_global_kernel_shap)

    tree_interp("global")
    output_path_global_tree_shap = os.path.join(
        output_path_global_dir, "shap_feature_importance_plots.html"
    )
    assert os.path.isfile(output_path_global_tree_shap)


def test_local_shap_tree() -> None:
    for files in os.listdir(output_path_local_dir):
        os.remove(os.path.join(output_path_local_dir, files))

    min_obs = 0
    max_obs = len(DATA_X)
    tree_interp("local")
    output_path_local_min_obs_shap = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{min_obs + 1}th_obs.html"
    )
    output_path_local_max_obs_shap = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{max_obs}th_obs.html"
    )
    outside_output = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{max_obs + 1}th_obs.html"
    )

    assert os.path.isfile(output_path_local_min_obs_shap)
    assert os.path.isfile(output_path_local_max_obs_shap)
    assert not os.path.isfile(outside_output)


def test_local_shap_kernel() -> None:
    for files in os.listdir(output_path_local_dir):
        os.remove(os.path.join(output_path_local_dir, files))

    min_obs = 0
    max_obs = len(DATA_X)
    kernel_interp("local")
    output_path_local_min_obs_shap = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{min_obs + 1}th_obs.html"
    )
    output_path_local_max_obs_shap = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{max_obs}th_obs.html"
    )
    outside_output = os.path.join(
        output_path_local_dir, f"shap_local_explanation_{max_obs + 1}th_obs.html"
    )

    assert os.path.isfile(output_path_local_min_obs_shap)
    assert os.path.isfile(output_path_local_max_obs_shap)
    assert not os.path.isfile(outside_output)
