# ML Base Package
Base classes and utilities that are useful for deploying ML models.

This package is useful for creating abstractions around machine learning models that make it easier to deploy them into 
other software systems. 

## Installation 

To download the code and set up a development use these instructions. 

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

## Running the Unit Tests
To run the unit test suite execute these commands:
```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```