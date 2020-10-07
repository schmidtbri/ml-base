from pydantic import BaseModel, Field
from pydantic import ValidationError
from enum import Enum

from ml_base.ml_model import MLModel, MLModelSchemaValidationException


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
class MLModelMock(MLModel):
    # accessing the package metadata
    display_name = "display_name"
    qualified_name = "qualified_name"
    description = "description"
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self):
        pass

    def predict(self, data):
        try:
            model_input = ModelInput(**data)
        except ValidationError as e:
            raise MLModelSchemaValidationException()
        return ModelOutput(species="Iris setosa")


# creating a mockup class to test with
class SomeClass(object):
    pass
