TEST_PATH=./tests

.DEFAULT_GOAL := help

.PHONY: help clean-pyc build clean-build venv dependencies test-dependencies clean-venv test test-reports clean-test check-codestyle check-docstyle

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

build: ## build a package
	python setup.py sdist bdist_wheel

clean-build:  ## clean build artifacts
	rm -rf build
	rm -rf dist
	rm -rf vendors
	rm -rf ml_base.egg-info

venv: ## create virtual environment
	python3 -m venv venv

dependencies: ## install dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install --upgrade setuptools
	python -m pip install --upgrade wheel
	pip install -r requirements.txt

test-dependencies: ## install dependencies from test_requirements.txt
	pip install -r test_requirements.txt

doc-dependencies: ## install dependencies from doc_requirements.txt
	pip install -r doc_requirements.txt

update-dependencies:  ## Update dependency versions
	pip-compile requirements.in > requirements.txt
	pip-compile test_requirements.in > test_requirements.txt
	pip-compile test_requirements.in > test_requirements.txt

clean-venv: ## remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y

test: clean-pyc ## Run unit test suite.
	pytest --verbose --color=yes $(TEST_PATH)

test-reports: clean-pyc clean-test ## Run unit test suite with reporting
	mkdir -p reports
	mkdir ./reports/unit_tests
	mkdir ./reports/coverage
	mkdir ./reports/badge
	python -m coverage run --source ml_base -m pytest --verbose --color=yes --html=./reports/unit_tests/report.html --junitxml=./reports/unit_tests/report.xml $(TEST_PATH)
	coverage html -d ./reports/coverage
	coverage-badge -o ./reports/badge/coverage.svg

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf reports
	rm -rf .pytype

check-codestyle:  ## checks the style of the code against PEP8
	pycodestyle ml_base --max-line-length=120

check-docstyle:  ## checks the style of the docstrings against PEP257
	pydocstyle ml_base

check-security:  ## checks for common security vulnerabilities
	bandit -r ml_base

check-dependencies:  ## checks for security vulnerabilities in dependencies
	safety check -r requirements.txt

check-codemetrics:  ## calculate code metrics of the package
	radon cc ml_base

check-pytype:  ## perform static code analysis
	pytype ml_base

build-docs:  ## build the documentation
	mkdocs build

view-docs:  ## open a web browser to view the documentation
	open site/index.html

clean-docs:  ## clean up the files in the docs build folder
	rm -rf site

run-examples:  ## run all example notebooks in examples directory
	jupyter nbconvert --to notebook --execute examples/basic.ipynb

convert-examples:  ## convert the example notebooks into Markdown files in docs folder
	jupyter nbconvert --to markdown examples/*.ipynb --output-dir='./docs'