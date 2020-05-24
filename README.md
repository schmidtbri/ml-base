# ML Base Package
Base classes and utilities that are useful for deploying ML models.

![Test and Build](https://github.com/schmidtbri/# ZeroRPC ML Model Deployment

![Test and Build](https://github.com/schmidtbri/ml-base/workflows/Test%20and%20Build/badge.svg)

## Introduction

This package is useful for creating abstractions around machine learning models that make it easier to deploy them into 
other software systems. 

## Examples
The main abstraction is the MLModel base class which wraps around the prediction functionality
of a model and exposes in an object oriented manner. Here is an example:

```python
from ml_base.ml_model import MLModel


class IrisModel(MLModel):
    def __init__(self):
        pass

    def predict(self):
        pass
```

The MLModel base class allows the developer to expose metadata about the model through object properties:

```python
class IrisModel(MLModel)
    self __init__(self):
        self.qualified_name
        self.display_name
        self.description
```

The MLModel base class also allows the developer to expose metadata about the model through class properties:

To expose schema information about the input and output of the model the Pydantic package is used:

```python
class IrisInput(Base):
    pass

class IrisOutput(Base):
    pass

class IrisModel(Base):
    input_schema = IrisInput

    output_schema = IrisOutput

```

## Installation 

### Installing from Pypi


### Installing from Source

### Installing for Development

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