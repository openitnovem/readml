import io
import os

from pkg_resources import parse_requirements
from setuptools import setup, find_packages

from version import __version__

DESCRIPTION = "Interpretability DS Package"
GIT_URL = "https://github.com/openitnovem/readml.git"
AUTHOR_EMAIL = "readml-dev@gmail.com"
REQUIREMENTS_PATH = os.path.join("requirements", "requirements.txt")
EXTRA_REQUIREMENTS_PATH=os.path.join("requirements", "extra-requirements.txt")

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

with io.open(REQUIREMENTS_PATH, encoding="utf-8") as f:
    install_requires = [str(requirement) for requirement in parse_requirements(f)]

def get_extra_requires(path, add_all=True):
    import re
    from collections import defaultdict
    with open(path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith('#'):
                tags = set()
                if '*:*' in k:
                    k, v = k.split('*:*')
                    tags.update(vv.strip() for vv in v.split(','))
                tags.add(re.split('[<=>]', k)[0])
                for t in tags:
                    extra_deps[t].add(k)
        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)
        print(extra_deps)
    return extra_deps


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
    extras_require=get_extra_requires(EXTRA_REQUIREMENTS_PATH), # inspired by https://hanxiao.io/2019/11/07/A-Better-Practice-for-Managing-extras-require-Dependencies-in-Python/
    python_requires=">=3.6.0",
)




