# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

"""
FBDTools library - icecream package
This module contains global configuration used by the icecream package
that can be changed at runtime
"""


class Options(object):
    """
    Configuration class that contains all configuration fields initiated to their default values
    """

    def __init__(self) -> None:
        self.default_number_bins = 15
        self.max_unicity_ratio = 0.05
        self.max_categories = 40
        self.max_recommended_categories = 80
        self.bars_color = "#1f77b4"
        self.predictions_color = "#ff7f0e"
        self.targets_color = "#2ca02c"
        self.special_color = "#d62728"
        self.heatmap_colorscale = "Reds"
        self.random_state = 4

    def __repr__(self) -> str:
        return "icecream configuration: {}".format(self.__dict__)


options = Options()
