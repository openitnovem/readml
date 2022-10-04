# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import os


def get_interp_env():
    return os.getenv("INTERP_ENV", "local")


env = get_interp_env()
