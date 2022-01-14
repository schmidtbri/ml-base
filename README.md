# ML Base Package

Base classes and utilities that are useful for deploying ML models.

This package is useful for creating abstractions around machine learning models that make it easier to deploy them into 
other software systems. 

## Installation 

The package can be installed from [pypi](https://pypi.org/project/ml-base/):

```bash
pip install ml_base
```

# Documentation

The package documentation is hosted [here](https://schmidtbri.github.io/ml-base/).


# Setting up a Development Environment

To download the source code execute this command:

```bash
git clone https://github.com/schmidtbri/ml-base
```

Then create a virtual environment and activate it:

```bash
# go into the project directory
cd ml-base

make venv

source venv/bin/activate
```

Install the dependencies:

```bash
make dependencies
```

The package uses [pip-tools](https://pypi.org/project/pip-tools/) to manage the dependencies. To install pip-tools execute this command:

```bash
pip install pip-tools
```

To update the dependencies in the requirements.txt file, execute this command:

```bash
pip-compile --upgrade
```

## Running the Tests

To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```