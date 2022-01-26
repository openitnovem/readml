import numpy as np
import pandas as pd
import pytest

from readml.icecream.check_utils import (
    check_ale,
    check_bins,
    check_clip_quantile,
    check_data_features,
    check_input_in_list,
    check_model_predict,
    check_sized_lengths,
)


class DummyModel(object):
    """
    Dummy class that acts like a scikit-learn supervised learning model
    """

    def __init__(self, classifier: bool = True, has_proba: bool = True) -> None:
        self.predict = lambda x: None
        if classifier:
            self.classes_ = [0, 1]
            if has_proba:
                self.predict_proba = lambda x: None


def test_check_clip_quantile() -> None:
    check_clip_quantile(0.0)
    with pytest.raises(Exception) as e_info:
        _ = check_clip_quantile(1.0)
    assert isinstance(e_info.value, ValueError)
    with pytest.raises(Exception) as e_info:
        _ = check_clip_quantile(0.5)
    assert isinstance(e_info.value, ValueError)


def test_check_model_predict() -> None:
    with pytest.raises(Exception) as e_info:
        _ = check_model_predict(4, False)
    assert isinstance(e_info.value, AttributeError)
    with pytest.raises(Exception) as e_info:
        _ = check_model_predict(4, True)
    assert isinstance(e_info.value, AttributeError)
    check_model_predict(DummyModel(False, False), False)
    check_model_predict(DummyModel(True, True), False)
    check_model_predict(DummyModel(True, True), True)
    with pytest.raises(Exception) as e_info:
        _ = check_model_predict(DummyModel(True, has_proba=False), True)
    assert isinstance(e_info.value, AttributeError)


def test_check_ale() -> None:
    data = pd.DataFrame({"a": [0, 1], "b": [2, 3], "c": ["a", "b"]})
    with pytest.raises(Exception) as e_info:
        _ = check_ale(data, ["a", "b", "c"], True)
    assert isinstance(e_info.value, NotImplementedError)


def test_check_data_features() -> None:
    data = pd.DataFrame({"a": [0, 1], "b": [2, 3]})
    check_data_features(data, ["a", "b"])
    with pytest.raises(Exception) as e_info:
        _ = check_data_features(data, ["c"])
    assert isinstance(e_info.value, ValueError)
    with pytest.raises(Exception) as e_info:
        _ = check_data_features(data, ["a", "c"])
    assert isinstance(e_info.value, ValueError)


def test_check_bins() -> None:
    check_bins(2)
    check_bins(np.int64(2))
    check_bins({"a": 2})
    check_bins({"a": [2, 3]})
    with pytest.raises(Exception) as e_info:
        _ = check_bins(2.0)
    assert isinstance(e_info.value, ValueError)
    with pytest.raises(Exception) as e_info:
        _ = check_bins([2])
    assert isinstance(e_info.value, ValueError)


def test_check_input_in_list() -> None:
    check_input_in_list("a", ["a", "b"])
    with pytest.raises(Exception) as e_info:
        _ = check_input_in_list("c", ["a", "b"])
    assert isinstance(e_info.value, ValueError)


def test_check_sized_lengths() -> None:
    check_sized_lengths(pd.Series([0, 2]), None)
    check_sized_lengths(pd.Series([0, 2]), pd.Series([1, 3]))
    check_sized_lengths(pd.Series([0, 2]), pd.Series([1, 3]), pd.Series([4, 5]))
    with pytest.raises(Exception) as e_info:
        _ = check_sized_lengths(pd.Series([0, 2]), pd.Series([3]))
    assert isinstance(e_info.value, ValueError)
    with pytest.raises(Exception) as e_info:
        _ = check_sized_lengths(pd.Series([0, 2]), pd.Series([4, 5]), pd.Series([3]))
    assert isinstance(e_info.value, ValueError)
