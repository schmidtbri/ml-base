from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ModelDetails(BaseModel):
    """High level details of a model."""
    display_name: str = Field(description="Display name of the model.")
    qualified_name: str = Field(description="Qualified name of the model.")
    description: str = Field(description="Description of the model.")
    version: str = Field(description="Code version of the model")
    model_parameters_version: Optional[str] = Field("Version of the parameters in the model object.")


class ModelParametersMetadata(BaseModel):
    """Details about a set of parameters of a model."""
    model_qualified_name: str = Field(description="Qualified name of the model to which these parameters belong.")
    model_version: str = Field(description="Model code version that these parameters belong to.")
    model_parameters_version: str = Field(description="Version of the model parameters.")
    creation_timestamp: datetime = Field(description="Datetime when the model parameters were created.")
    description: Optional[str] = Field(description="Short description for the model parameters.")
    author: Optional[str] = Field(description="Name of person who created the model parameters.")
    author_email: Optional[str] = Field(description="Email of person who created the model parameters.")
    tags: Optional[List[str]] = Field(description="List of strings that hold information about the parameters.")
    metadata: Optional[Dict[str, Any]] = Field(description="Key value pairs containing any extra metadata about the "
                                                           "model parameters.")
    dependencies: Optional[List[str]] = Field(description="List of code dependencies.")


class ModelMetadata(ModelDetails):
    """Low level details of a model."""
    # TODO: add model for JSON schema
    parameters: List[ModelParametersMetadata] = Field(description="Parameter metadata about the parameters available "
                                                                  "for the model.")
    input_schema: str = Field(description="Input schema for the model.")
    output_schema: str = Field(description="Output schema for the model.")
