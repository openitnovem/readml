import os

from readml.logger import logger


def initialize_dir(out_path: str):
    """
    Check if output dir tree exists if not it will be created.
    Output dir includes global_interpretation and local_interpretation folders

    Parameters
    ----------
    out_path : str
        outputs path, folder where interpretation reports will be saved
    """
    log_msg = "Directory created"
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        logger.info("%s %s ", log_msg, out_path)
    if not os.path.isdir(os.path.join(out_path, "local_interpretation")):
        os.makedirs(os.path.join(out_path, "local_interpretation"))
        logger.info(
            "%s %s ",
            log_msg,
            os.path.join(out_path, "local_interpretation"),
        )
    if not os.path.isdir(os.path.join(out_path, "global_interpretation")):
        os.makedirs(os.path.join(out_path, "global_interpretation"))
        logger.info(
            "%s %s ",
            log_msg,
            os.path.join(out_path, "global_interpretation"),
        )
