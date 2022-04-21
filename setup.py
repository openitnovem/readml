import io
import os

from pkg_resources import parse_requirements
from setuptools import setup, find_packages

from version import __version__

DESCRIPTION = "Interpretability DS Package"
GIT_URL = "https://github.com/openitnovem/readml.git"
AUTHOR_EMAIL = "readml-dev@gmail.com"
REQUIREMENTS_PATH = os.path.join("requirements", "requirements_m1.txt")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

conditional_requires = ["pickle5==0.0.11; python_version < '3.8'"] # Pattern : package_name==0.0.00; python_version >= X.X
with io.open(REQUIREMENTS_PATH, encoding="utf-8") as f:
    install_requires = [str(requirement) for requirement in parse_requirements(f)]
    install_requires = conditional_requires + install_requires

setup(
    name="readml",
    packages=find_packages(),
    version=__version__,
    description="This project aims to interpret ML & DL models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GIT_URL,
    author="Factory D&IA",
    author_email=AUTHOR_EMAIL,
    license="Apache Software License 2.0",
    install_requires=install_requires,
    python_requires=">=3.6.0",
)
