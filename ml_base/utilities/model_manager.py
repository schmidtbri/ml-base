"""Model Manager class for loading, managing, and interacting with models."""
import importlib
from typing import List

from ml_base.ml_model import MLModel


class ModelManager(object):
    """Singleton class that instantiates and manages model objects."""

    def __new__(cls):  # noqa: D102
        if not hasattr(cls, "instance"):
            cls.instance = super(ModelManager, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        """Construct ModelManager object."""
        self._models = []

    def load_model(self, class_path: str) -> None:
        """Import and instantiate an MLModel object from a class path.

        :param class_path: Class path to the model's MLModel class.
        :type class_path: str
        :raises ValueError: Raised if the model is not a subtype of MLModel, or if a model with the same qualified name
          is already loaded in the ModelManager.
        :rtype: None

        """
        # splitting the class_path into module path and class name
        module_path = ".".join(class_path.split(".")[:-1])
        class_name = class_path.split(".")[-1]

        # importing the model class
        model_module = importlib.import_module(module_path)
        model_class = getattr(model_module, class_name)

        # instantiating the model object from the class
        model_object = model_class()

        if not isinstance(model_object, MLModel):
            raise ValueError("ModelManager instance can only hold references to objects of type MLModel.")

        if model_object.qualified_name in [model.qualified_name for model in self._models]:
            raise ValueError("A model with the same qualified name is already in the ModelManager singleton.")

        # saving the model reference to the models list
        self._models.append(model_object)

    def remove_model(self, qualified_name: str) -> None:
        """Remove an MLModel object from the ModelManager singleton.

        :param qualified_name: The qualified name of the model to be returned.
        :type qualified_name: str
        :raises ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.
        :return: None
        :rtype: None

        """
        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            self._models.remove(model_objects[0])

    def get_models(self) -> List[dict]:
        """Get a list of models in the model manager singleton.

        :return: List of dictionaries containing information about the model instances in the ModelManager singleton.
        :rtype: list

        .. note::
            The dictionaries in the list returned by this method contain these keys:

            - display_name
            - qualified_name
            - description
            - version

        """
        model_objects = [{"display_name": model.display_name,
                          "qualified_name": model.qualified_name,
                          "description": model.description,
                          "version": model.version} for model in self._models]
        return model_objects

    def get_model_metadata(self, qualified_name: str) -> dict:
        """Get model metadata by qualified name.

        :param qualified_name:
        :type qualified_name:
        :return: Dictionary containing information about a model in the ModelManager singleton.
        :rtype: dict

        .. note::
            The dictionaries in the list returned by this method contain these keys:

            - display_name
            - qualified_name
            - description
            - version
            - input_schema
            - output_schema

        """
        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            model_object = model_objects[0]
            return {
                "display_name": model_object.display_name,
                "qualified_name": model_object.qualified_name,
                "description": model_object.description,
                "version": model_object.version,
                "input_schema": model_object.input_schema.schema(),
                "output_schema": model_object.output_schema.schema()
            }

    def get_model(self, qualified_name: str) -> MLModel:
        """Get a model object by qualified name.

        :param qualified_name: The qualified name of the model to be returned.
        :type qualified_name: str
        :raises ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.
        :return: Model object
        :rtype: MLModel

        """
        # searching the list of model objects to find the one with the right qualified name
        model_objects = [model for model in self._models if model.qualified_name == qualified_name]

        if len(model_objects) == 0:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            return model_objects[0]
