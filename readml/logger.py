# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import colorlog

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_logger():

    # Create handlers
    log_directory = os.path.join(ROOT_DIR, "../outputs")
    log_path = os.path.join(log_directory, "logs.log")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    f_handler = logging.FileHandler(log_path)

    log_format = "%(asctime)s --- %(name)s --- %(levelname)s --- %(message)s"
    bold_seq = "\033[1m"
    colorlog_format = f"{bold_seq} " "%(log_color)s " f"{log_format}"
    handler_color = colorlog.StreamHandler()
    handler_color.setFormatter(colorlog.ColoredFormatter(colorlog_format))

    # Create formatters and add it to handlers
    f_format = logging.Formatter(log_format)
    f_handler.setFormatter(f_format)

    # Create a custom logger (log)
    log = logging.getLogger("readml")
    log.setLevel(logging.DEBUG)

    # Add handlers to the logger (log)
    log.addHandler(f_handler)
    log.addHandler(handler_color)
    return log


logger = get_logger()
