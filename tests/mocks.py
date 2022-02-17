from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, create_model

from ml_base.ml_model import MLModel, MLModelException
from ml_base.decorator import MLModelDecorator


class ModelInput(BaseModel):
    sepal_length: float = Field(gt=5.0, lt=8.0)
    sepal_width: float = Field(gt=2.0, lt=6.0)
    petal_length: float = Field(gt=1.0, lt=6.8)
    petal_width: float = Field(gt=0.0, lt=3.0)


class Species(str, Enum):
    iris_setosa = "Iris setosa"
    iris_versicolor = "Iris versicolor"
    iris_virginica = "Iris virginica"


class ModelOutput(BaseModel):
    species: Species


# creating an MLModel class to test with
class IrisModelMock(MLModel):
    display_name = "display_name"
    qualified_name = "qualified_name"
    description = "description"
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self):
        pass

    def predict(self, data: ModelInput) -> ModelOutput:
        return ModelOutput(species="Iris setosa")


class IrisModelMockThatRaisesException(MLModel):
    display_name = "display_name"
    qualified_name = "qualified_name"
    description = "description"
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self):
        pass

    def predict(self, data: ModelInput) -> ModelOutput:
        raise MLModelException("Exception!")


# creating a mockup class to test with
class SomeClass(object):
    pass


class SimpleDecorator(MLModelDecorator):
    """Decorator that does nothing."""
    pass


class AddStringDecorator(MLModelDecorator):
    """Decorator that adds a string to the display_name, qualified_name, description, and version string returned
    by the model object."""

    @property
    def display_name(self) -> str:
        return self._model.display_name + self._configuration["string"]

    @property
    def qualified_name(self) -> str:
        return self._model.qualified_name + self._configuration["string"]

    @property
    def description(self) -> str:
        return self._model.description + self._configuration["string"]

    @property
    def version(self) -> str:
        return self._model.version + self._configuration["string"]


class CatchExceptionsDecorator(MLModelDecorator):
    """Decorator that catches exceptions thrown by the predict method of the model and raises an MLModelException
    instead."""

    def predict(self, data):
        try:
            return self._model.predict(data=data)
        except Exception as e:
            raise MLModelException(e)


class ModifySchemaDecorator(MLModelDecorator):
    """Decorator that modifies the input and output schemas of the model object."""

    @property
    def input_schema(self):
        input_schema = self._model.input_schema
        new_input_schema = create_model(
            input_schema.__name__,
            correlation_id=(UUID, ...),
            __base__=input_schema,
        )
        return new_input_schema

    @property
    def output_schema(self):
        output_schema = self._model.output_schema
        new_output_schema = create_model(
            output_schema.__name__,
            correlation_id=(UUID, ...),
            __base__=output_schema,
        )
        return new_output_schema

    def predict(self, data):
        prediction = self._model.predict(data=data)
        wrapped_prediction = {
            "correlation_id": uuid4(),
            "model_input": prediction
        }
        return wrapped_prediction
