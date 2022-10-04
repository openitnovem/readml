# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

this_abs_path = os.path.dirname(os.path.abspath(__file__))
meta_file_path = os.path.join(this_abs_path, "meta_build.json")

"""
Read meta_build.json file and add version value to global variable __version__
"""

try:
    with open(meta_file_path) as jfd:
        datas = jfd.read()
    json_datas = json.loads(datas)
    __version__ = json_datas["version"]
except FileNotFoundError:
    __version__ = "0.0.0.dev0"
