![Code Quality Status](https://github.com/schmidtbri/ml-base/actions/workflows/test.yml/badge.svg)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPi](https://img.shields.io/badge/pypi-v0.2.1-green)](https://pypi.org/project/ml-base/)

# ml-base

**ml-base** is a package that provides base classes and utilities that are useful for deploying machine learning models.

## Installation 

The easiest way to install ml-base is using pip

```bash
pip install ml-base
```

## Usage

There are several examples of how to use the ml-base framework in the 
[documentation](https://schmidtbri.github.io/ml-base/).

## Development

First, download the source code with this command:

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

## Testing 

To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```
