"""Base class for building ML models that are easy to deploy and integrate."""
from abc import ABC, abstractmethod
from typing import List, Optional

from ml_base.schemas import ModelParametersMetadata


class MLModel(ABC):
    """Base class for ML model prediction code."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Abstract property that returns a display name for the model.

        Returns:
            str: The display name of the model.

        !!! note
            This is a name for the model that looks good in user interfaces.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def qualified_name(self) -> str:
        """Abstract property that returns the qualified name of the model.

        Returns:
            str: The qualified name of the model.

        !!! warning
            A qualified name is an unambiguous identifier for the model.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def description(self) -> str:
        """Abstract property that returns a description of the model.

        Returns:
            str: The description of the model.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def version(self) -> str:
        """Abstract property that returns the model's version as a string.

        Returns:
            str: The version of the model.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_schema(self):
        """Property that returns the schema type that is accepted by the predict() method.

        Returns:
            pydantic.BaseModel: The input schema of the model.

        !!! note
            This property must return a subtype of pydantic.BaseModel.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_schema(self):
        """Property returns the schema type that is returned by the predict() method.

        Returns:
            pydantic.BaseModel: The output schema of the model.

        !!! note
            This property must return a subtype of pydantic.BaseModel.

        """
        raise NotImplementedError()

    @classmethod
    def parameters(cls) -> List[ModelParametersMetadata]:
        """Return a list of ModelParameterMetadata objects that describe the parameters available for use.

        !!! warning
            This class method returns an empty list by default. It needs to be overridden by any classes that inherit
            from MLModel to return information about the parameters available for the model.

        """
        return []

    @property
    def parameters_metadata(self) -> Optional[ModelParametersMetadata]:
        """Return the metadata of the parameters currently loaded in the model object.

        !!! warning
            This property returns None by default. It needs to be overridden by any classes that inherit from MLModel
            that are able to return information about the parameters currently loaded in the model object.

        """
        return None

    @abstractmethod
    def __init__(self, model_parameters_version: Optional[str] = None) -> None:
        """Create an MLModel instance by adding any deserialization and initialization code for the model."""
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        """Prediction with the model.

        Args:
            data: Data used by the model for making a prediction

        Returns:
            Prediction

        """
        raise NotImplementedError()


class MLModelException(Exception):
    """Exception base class used to raise exceptions within MLModel derived classes."""

    def __init__(self, *args):
        """Initialize MLModelException instance."""
        Exception.__init__(self, *args)


class MLModelParametersNotAvailableException(MLModelException):
    """Exception type raised when requested model parameters are not available."""

    def __init__(self, *args):
        """Initialize MLModelParametersNotAvailable instance."""
        MLModelException.__init__(self, *args)


class MLModelSchemaValidationException(MLModelException):
    """Exception type used to raise schema validation exceptions within MLModel derived classes."""

    def __init__(self, *args):
        """Initialize MLModelSchemaValidationException instance."""
        MLModelException.__init__(self, *args)
