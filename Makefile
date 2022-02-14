# Guard against running Make commands outside a virtualenv or conda env
venv:
ifndef VIRTUAL_ENV
ifndef CONDA_PREFIX
$(error VIRTUAL / CONDA ENV is not set - please activate environment)
endif
endif

clean: venv
	@echo "Removing build artifacts / temp files"
	find . -name "*.pyc" -delete

deps: venv
	pip install -Ue .

