"""Model Manager class for loading, managing, and interacting with models."""
import importlib
from typing import List
from threading import Lock

from ml_base import MLModel, MLModelDecorator


class ModelManager(object):
    """Singleton class that instantiates and manages model objects."""

    _lock = Lock()

    def __new__(cls) -> "ModelManager":  # noqa: D102
        """Create and return a new ModelManager instance, after instance is first created it will always be returned."""
        if not hasattr(cls, "_instance"):
            with cls._lock:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._is_initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Construct ModelManager object."""
        if self._is_initialized is False:  # pytype: disable=attribute-error
            self._models = []
            self._is_initialized = True

    @classmethod
    def clear_instance(cls) -> None:
        """Clear singleton instance from class."""
        del cls._instance

    def load_model(self, class_path: str) -> None:
        """Import and instantiate an MLModel object from a class path.

        Args:
            class_path: Class path to the model's MLModel class.

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
        model_object = model_class()

        self.add_model(model_object)

    def add_model(self, model: MLModel) -> None:
        """Add a model to the ModelManager.

        Args:
            model: instance of MLModel

        """
        if not isinstance(model, MLModel):
            raise ValueError("ModelManager instance can only hold references to objects of type MLModel.")

        if model.qualified_name in [model.qualified_name for model in self._models]:
            raise ValueError("A model with the same qualified name is already in the ModelManager singleton.")

        # saving the model reference to the models list
        self._models.append(model)

    def remove_model(self, qualified_name: str) -> None:
        """Remove an MLModel object from the ModelManager singleton.

        Args:
            qualified_name: The qualified name of the model to be returned.

        Raises:
            ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.

        """
        # searching the list of model objects to find the one with the right qualified name
        model = next((model for model in self._models if model.qualified_name == qualified_name), None)

        if model is None:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            self._models.remove(model)

    def get_models(self) -> List[dict]:
        """Get a list of models in the model manager singleton.

        Returns:
            List of dictionaries containing information about the model instances in the ModelManager singleton.

        !!! note
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

        Args:
            qualified_name: Qualified name of the model for which to get metadata

        Returns:
            Dictionary containing information about a model in the ModelManager singleton.

        !!! note
            The dictionaries in the list returned by this method contain these keys:

            - display_name
            - qualified_name
            - description
            - version
            - input_schema
            - output_schema

        """
        # searching the list of model objects to find the one with the right qualified name
        model = next((model for model in self._models if model.qualified_name == qualified_name), None)

        if model is None:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            return {
                "display_name": model.display_name,
                "qualified_name": model.qualified_name,
                "description": model.description,
                "version": model.version,
                "input_schema": model.input_schema.schema(),
                "output_schema": model.output_schema.schema()
            }

    def get_model(self, qualified_name: str) -> MLModel:
        """Get a model object by qualified name.

        Args:
            qualified_name: The qualified name of the model to be returned.

        Returns:
            Model object

        Raises:
            ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.

        """
        # searching the list of model objects to find the one with the right qualified name
        model = next((model for model in self._models if model.qualified_name == qualified_name), None)

        if model is None:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))
        else:
            return model

    def add_decorator(self, qualified_name: str, decorator: MLModelDecorator) -> None:
        """Add a decorator to a model object by qualified name.

        Args:
            qualified_name: The qualified name of the model to add decorator to.
            decorator: MLModelDecorator instance to apply to model instance.

        Returns:
            None

        Raises:
            ValueError: Raised if a model with the qualified name can't be found in the ModelManager singleton.

        """
        # searching the list of model objects to find the one with the right qualified name
        model = next((model for model in self._models if model.qualified_name == qualified_name), None)

        if model is None:
            raise ValueError("Instance of model '{}' not found in ModelManager.".format(qualified_name))

        # removing old model reference
        self.remove_model(qualified_name)

        # adding the decorator to the model object
        decorated_model = decorator.set_model(model)

        # adding the decorated model to the collection
        self.add_model(decorated_model)
