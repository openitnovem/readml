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
	isort -rc fbd_interpreter/ tests/
	black fbd_interpreter/ tests/

check:
	flake8 fbd_interpreter/ tests/
	mypy .

convert_notebooks:
	find notebooks/ -type f -name "*.ipynb" -not -path "*ipynb_checkpoints*" -exec jupytext --to py:percent --pipe black {} \;

test:
	python -m pytest tests/

build_egg:
	python setup.py bdist_egg