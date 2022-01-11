import os

from readml.logger import ROOT_DIR
from readml.resource.output_builders import initialize_dir


def test_initialize_dir():
    out_path = os.path.join(ROOT_DIR, "../outputs/tests")
    initialize_dir(out_path)
    assert os.path.isdir(out_path)
    assert os.path.isdir(os.path.join(out_path, "global_interpretation"))
    assert os.path.isdir(os.path.join(out_path, "local_interpretation"))
