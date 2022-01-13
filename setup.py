import io
import os

from pkg_resources import parse_requirements
from setuptools import setup, find_packages

from version import __version__

DESCRIPTION = "Interpretability DS Package"
GIT_URL = (
    "https://gitlab-repo-gpf.apps.eul.sncf.fr/digital/"
    "groupefbd-digital/90023/DSE/fbdtools-python/fbd-interpreter.git"
)
REQUIREMENTS_PATH = os.path.join("requirements", "requirements.txt")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

with io.open(REQUIREMENTS_PATH, encoding="utf-8") as f:
    install_requires = [str(requirement) for requirement in parse_requirements(f)]

setup(
    name="readml",
    packages=find_packages(),
    version=__version__,
    description="This project aims to interpret ML & DL models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GIT_URL,
    author="Soumaya IHIHI - DS",
    author_email="",
    license="Apache Software License 2.0",
    install_requires=install_requires,
    python_requires=">=3.6.0",
)
