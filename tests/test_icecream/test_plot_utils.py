# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from readml.icecream.plot_utils import detect_axis_range


def test_detect_axis_range() -> None:
    assert detect_axis_range(None) is None
    assert detect_axis_range(pd.Series([])) is None
    assert detect_axis_range(pd.Series(["a", "b"])) is None
    assert detect_axis_range(pd.Series([0, 1]), pd.Series([0.5, 0.5])) == [-0.05, 1.05]
    assert detect_axis_range(pd.Series([-1, 1]), pd.Series([0, 1])) is None
