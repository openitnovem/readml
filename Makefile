# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

REQUIREMENTSDIR = requirements

install:
	pip install pip-tools
	pip-sync $(REQUIREMENTSDIR)/requirements.txt \
		#$(REQUIREMENTSDIR)/requirements-dev.txt

set_git_hooks:
	cp -r git_hooks/* .git/hooks/

upgrade:
	pip-compile $(REQUIREMENTSDIR)/requirements.in
	pip-compile $(REQUIREMENTSDIR)/requirements-dev.in
	pip-sync $(REQUIREMENTSDIR)/requirements.txt \
		$(REQUIREMENTSDIR)/requirements-dev.txt \

format:
	isort -rc readml/ tests/
	black readml/ tests/

check:
	flake8 readml/ tests/
	mypy .

convert_notebooks:
	find notebooks/ -type f -name "*.ipynb" -not -path "*ipynb_checkpoints*" -exec jupytext --to py:percent --pipe black {} \;

test:
	python -m pytest tests/

build_egg:
	python setup.py bdist_egg