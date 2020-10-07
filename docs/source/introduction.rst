************
Introduction
************

The ml_base package defines a common set of base classes that are useful for working with machine learning model
prediction code. The base classes define a set of interfaces that help to write ML code that is reusable and testable.

The core of the ml_base package is the MLModel class which defines a simple interface for doing machine learning model
prediction. The MLModel base class allows the developer of the model to return this information to the user of the
model in a standardized way:

- Model Qualified Name, a unique identifier for the model
- Model Display Name, a friendly name for the model used in user interfaces
- Model Description, a description for the model
- Model Version, semantic version of the model codebase
- Model Input Schema, an object that describes the model's input data
- Model Output Schema, an object that describes the model's output schema

The package also includes a ModelManager class that is able to instantiate and manage models that are created using the
MLModel base class.


FAQ
###

* Why bother with base classes and interfaces? Isn't it just extra work?

Interface-driven software development can be very helpful when building complex software systems. By using the MLModel
base class to deliver the prediction functionality, developing software that makes use of the machine learning model is
greatly simplified and the model is much more accesible and easier to use. Developing the prediction functionality of
your ML model around the MLModel base class provides a simple "meeting point" between your model and anyone that wants
to use it, the user doesn't need to worry about the implementation of the model and you don't need to worry about the
use cases that your model will be used in.

* Why not just deliver a serialized model object to the software engineer?

Having a class that wraps around your model object provides a great place to do things that make your model easier
to use. For example:

    - Deserialize model parameters from disk so that using the model is a easy as instantiating a class and calling
      predict()
    - Validate inputs before sending them to the model
    - Modify predictions before sending them back to the calling code
    - Return metadata about your model
    - Convert model inputs from a developer-friendly data structure (dictionaries and lists) to a model-friendly data
      structure (dataframes)
    - Convert model outputs from a dataframe to a dictionary or list


* So what do I have to do to use the base classes?

Create a wrapper class around your model that inherits from the MLModel base class and implement the required methods.
You can follow the example implementation available in the documentation.
