TEST_PATH=./tests

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

clean-pyc: ## Remove python artifacts.
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
.PHONY: clean-pyc

build: ## Build the package
	python setup.py sdist bdist_wheel
.PHONY: build

clean-build:  ## Clean build artifacts
	rm -rf build
	rm -rf dist
	rm -rf vendors
	rm -rf ml_base.egg-info
.PHONY: clean-build

venv: ## Create virtual environment
	python3 -m venv venv
.PHONY: venv

dependencies: ## Install dependencies from requirements.txt
	python -m pip install --upgrade pip
	python -m pip install --upgrade setuptools
	python -m pip install --upgrade wheel
	pip install -r requirements.txt
.PHONY: dependencies

test-dependencies: ## Install dependencies from test_requirements.txt
	pip install -r test_requirements.txt
.PHONY: test-dependencies

doc-dependencies: ## Install dependencies from doc_requirements.txt
	pip install -r doc_requirements.txt
.PHONY: doc-dependencies

update-dependencies:  ## Update dependency versions
	pip-compile requirements.in > requirements.txt
	pip-compile test_requirements.in > test_requirements.txt
	pip-compile doc_requirements.in > doc_requirements.txt
.PHONY: update-dependencies

clean-venv: ## Remove all packages from virtual environment
	pip freeze | grep -v "^-e" | xargs pip uninstall -y
.PHONY: clean-venv

test: clean-pyc ## Run unit test suite.
	pytest --verbose --color=yes $(TEST_PATH)
.PHONY: test

test-reports: clean-pyc clean-test ## Run unit test suite with reporting
	mkdir -p reports
	mkdir ./reports/unit_tests
	mkdir ./reports/coverage
	mkdir ./reports/badge
	python -m coverage run --source ml_base -m pytest --verbose --color=yes --html=./reports/unit_tests/report.html --junitxml=./reports/unit_tests/report.xml $(TEST_PATH)
	coverage html -d ./reports/coverage
	coverage-badge -o ./reports/badge/coverage.svg
.PHONY: test-reports

clean-test:	## Remove test artifacts
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf reports
	rm -rf .pytype
.PHONY: clean-test

check-codestyle:  ## Check the style of the code against PEP8
	pycodestyle ml_base --max-line-length=120
.PHONY: check-codestyle

check-docstyle:  ## Check the style of the docstrings against PEP257
	pydocstyle ml_base
.PHONY: check-docstyle

check-security:  ## Checks for common security vulnerabilities
	bandit -r ml_base
.PHONY: check-security

check-dependencies:  ## Check for security vulnerabilities in dependencies
	safety check -r requirements.txt
.PHONY: check-dependencies

check-codemetrics:  ## Calculate code metrics of the package
	radon cc ml_base
.PHONY: check-codemetrics

check-annotations: ## Check for type annotations coverage
	flake8 ml_base --max-line-length=120 --ignore=ANN101,ANN102,ANN401
.PHONY: check-annotations

build-docs:  ## Build the documentation
	mkdocs build
.PHONY:build-docs

view-docs:  ## Open a web browser to view the documentation
	open site/index.html
.PHONY: view-docs

clean-docs:  ## Clean up the files in the docs build folder
	rm -rf site
.PHONY: clean-docs

run-examples:  ## Run all example notebooks in examples directory
	jupyter nbconvert --to notebook --execute examples/basic.ipynb
.PHONY: run-examples

convert-examples:  ## Convert the example notebooks into Markdown files in docs folder
	jupyter nbconvert --to markdown examples/*.ipynb --output-dir='./docs'
.PHONY:convert-examples
