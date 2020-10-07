"""Base class for building ML models that are easy to deploy and integrate."""
from abc import ABC, abstractmethod


class MLModel(ABC):
    """Base class for ML model prediction code."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Abstract property that returns a display name for the model.

        .. note::
            This is a name for the model that looks good in user interfaces.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def qualified_name(self) -> str:
        """Abstract property that returns the qualified name of the model.

        .. note::
            A qualified name is an unambiguous identifier for the model.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def description(self) -> str:
        """Abstract property that returns a description of the model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def version(self) -> str:
        """Abstract property that returns the model's version as a string."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_schema(self):
        """Property that returns the schema that is accepted by the predict() method.

        .. note::
            This property must return a subtype of pydantic.BaseModel.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_schema(self):
        """Property returns the schema that is returned by the predict() method.

        .. note::
            This property must return a subtype of pydantic.BaseModel.

        """
        raise NotImplementedError()

    @abstractmethod
    def __init__(self):
        """Create an MLModel instance by adding any deserialization and initialization code for the model."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        """Prediction with the model.

        :param data: data used by the model for making a prediction
        :type data: object --  can be any python type
        :rtype: python object -- can be any python type

        """
        raise NotImplementedError()


class MLModelException(Exception):
    """Exception base class used to raise exceptions within MLModel derived classes."""

    def __init__(self, *args):
        """Initialize MLModelException instance."""
        Exception.__init__(self, *args)


class MLModelSchemaValidationException(MLModelException):
    """Exception type used to raise schema validation exceptions within MLModel derived classes."""

    def __init__(self, *args):
        """Initialize MLModelSchemaValidationException instance."""
        MLModelException.__init__(self, *args)
