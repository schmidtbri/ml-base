from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

from ml_base.ml_model import MLModel, MLModelSchemaValidationException
from ml_base.schemas import ModelParametersMetadata


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
class MLModelMockWithoutParametersMetadata(MLModel):
    # accessing the package metadata
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


class MLModelMockWithParametersMetadata(MLModel):
    display_name = "display_name"
    qualified_name = "qualified_name"
    description = "description"
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    @classmethod
    def parameters(cls) -> List[ModelParametersMetadata]:
        model_parameters_metadata = ModelParametersMetadata(
            model_qualified_name="display_name",
            model_version="0.1.0",
            model_parameters_version="1",
            description="description",
            creation_timestamp=datetime.utcnow(),
            author="",
            author_email="",
            metadata={},
            dependencies=["", ""])

        return [model_parameters_metadata]

    @property
    def parameters_metadata(self) -> Optional[ModelParametersMetadata]:
        return ModelParametersMetadata(
            model_qualified_name="display_name",
            model_version="0.1.0",
            model_parameters_version="1",
            description="description",
            creation_timestamp=datetime.utcnow(),
            author="",
            author_email="",
            metadata={},
            dependencies=["", ""])

    def __init__(self):
        pass

    def predict(self, data: ModelInput) -> ModelOutput:
        return ModelOutput(species="Iris setosa")


# creating a mock class to test with
class SomeClass(object):
    pass
