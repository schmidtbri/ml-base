"""Model Manager class for loading, managing, and interacting with models."""
import importlib
from typing import List, Dict, Union, Type
from collections import defaultdict

from ml_base import MLModel
from ml_base.schemas import ModelDetails, ModelMetadata


class ModelManager(object):
    """Singleton class that instantiates and manages model objects."""

    def __new__(cls):  # noqa: D102
        """Create and return a new ModelManager instance, after instance is first created it will always be returned."""
        if not hasattr(cls, "_instance"):
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._is_initialized = False
        return cls._instance

    def __init__(self):
        """Construct ModelManager object."""
        if self._is_initialized is False:  # pytype: disable=attribute-error
            self._models: Dict[Type[MLModel], Union[MLModel, List[MLModel]]] = {}
            self._is_initialized = True

    @classmethod
    def clear_instance(cls):
        """Clear singleton instance from class."""
        del cls._instance

    def load_model(self, class_path: str, model_parameters_version: str = None) -> None:
        """Import and instantiate an MLModel object from a class path.

        Args:
            class_path: Class path to the model's MLModel class.
            model_parameters_version: version of the parameters to load, this parameter is passed to the __init__
            method of the model class.

        Raises:
            ValueError: Raised if the model is not a subtype of MLModel, or if a model with the same qualified name
            is already loaded in the ModelManager.

        """
        # splitting the class_path into module path and class name
        module_path = ".".join(class_path.split(".")[:-1])
        class_name = class_path.split(".")[-1]

        # importing the model class
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)

        # instantiating the model object from the class
        if model_parameters_version is None:
            model_object = model_class()
        else:
            model_object = model_class(model_parameters_version=model_parameters_version)

        self.add_model(model_object)

    def add_model(self, model: MLModel) -> None:
        """Add a model to the ModelManager.

        Args:
            model: instance of MLModel

        """
        if not isinstance(model, MLModel):
            raise ValueError("ModelManager instance can only hold references to objects of type MLModel.")

        # saving the model reference to the models list
        # if the model object does not provide parameters metadata
        if model.parameters_metadata is None:
            # save the model object directly to the dict since we will never hold more than one instance of the model
            self._models[type(model)] = model
        # if the model object provides parameters metadata
        else:
            if type(model) not in self._models.keys():
                self._models[type(model)] = []
            # check to see if the parameters version is already in the list
            model_parameters_version = model.parameters_metadata.model_parameters_version
            if any(True if inner_model.parameters_metadata.model_parameters_version == model_parameters_version
                   else False for inner_model in self._models[type(model)]):
                raise ValueError("An instance of {} with this parameters version is already loaded in "
                                 "the ModelManager.")
            else:
                # append the model object to a list since we expect to hold other instances of the model with
                # different parameter versions
                self._models[type(model)].append(model)

    def remove_model(self, qualified_name: str, parameters_version: str = None) -> None:
        """Remove an MLModel object from the ModelManager singleton.

        Args:
            qualified_name: The qualified name of the model to be returned.
            parameters_version: Version of the parameters to remove, if not provided all parameters versions are
            removed.

        Raises:
            ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.

        """
        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            self._models.remove(model_objects[0])

    def get_model_details(self) -> List[ModelDetails]:
        """Get details of all of the models in the model manager singleton.

        Returns:
            List of ModelDetails objects containing information about the model instances in the ModelManager singleton.

        """
        model_details = []
        for model_type, model_instances in self._models.items():
            if type(model_instances) is list:
                for model_instance in model_instances:
                    model_details.append(ModelDetails(
                        display_name=model_instance.display_name,
                        qualified_name=model_instance.qualified_name,
                        description=model_instance.description,
                        version=model_instance.version,
                        model_parameters_version=model_instance.parameters_metadata.model_parameters_version)
            else:
                model_details.append(ModelDetails(
                    display_name=model_instance.display_name,
                    qualified_name=model_instance.qualified_name,
                    description=model_instance.description,
                    version=model_instance.version,
                    model_parameters_version=model_instance.parameters_metadata.model_parameters_version)
        return model_objects

    def get_model_metadata(self, qualified_name: str) -> ModelMetadata:
        """Get model metadata by qualified name.

        Args:
            qualified_name: Qualified name of the model for which to get metadata

        Returns:
            ModelMetadata object with the metadata available for the model selected.

        """
        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            model_object = model_objects[0]
            return ModelMetadata(**{
                "display_name": model_object.display_name,
                "qualified_name": model_object.qualified_name,
                "description": model_object.description,
                "version": model_object.version,
                "parameters": model_object.parameters(),
                "input_schema": model_object.input_schema.schema(),
                "output_schema": model_object.output_schema.schema()
            })

    def get_model(self, qualified_name: str, parameters_version: str = None) -> MLModel:
        """Get a model object by qualified name.

        Args:
            qualified_name: The qualified name of the model to be returned.
            parameters_version: The parameters version, if not provided the latest parameters will be selected.

        Returns:
            Model object.

        Raises:
            ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.

        """
        # TODO: handle case when parameters_version is not provided

        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            return model_objects[0]
