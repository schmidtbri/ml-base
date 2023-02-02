"""Base class for building decorators for MLModel objects."""
from typing import Optional, Any
from ml_base.ml_model import MLModel
from pydantic import BaseModel


class MLModelDecorator(MLModel):
    """Base class for ML model decorator code.

    !!! note
        The default behavior of the MLModelDecorator base class is to do nothing and to forward the method call to
        the model that is is wrapping. Any subtypes of MLModelDecorator that would like to add on to the behavior
        of the model needs to override the default implementations in the MLModelDecorator base class.

    """

    _decorator_attributes = ["_model", "_configuration"]

    def __init__(self, model: Optional[MLModel] = None, **kwargs: dict) -> None:
        """Initialize MLModelDecorator instance.

        !!! note
            The MLModel parameter is optional and does not need to be provided at initialization of the decorator
            instance.

        !!! note
            This method receives the model instance and stores the reference.

        """
        if model is not None and not isinstance(model, MLModel):
            raise ValueError("Only objects of type MLModel can be wrapped with MLModelDecorator instances.")

        self.__dict__["_model"] = model
        self.__dict__["_configuration"] = kwargs

    def __repr__(self) -> str:
        """Return a string describing the decorator and the model that it is decorating."""
        return "{}({})".format(self.__class__.__name__, str(self.__dict__["_model"]))

    def set_model(self, model: MLModel) -> "MLModelDecorator":
        """Set a model in the decorator instance."""
        if not isinstance(model, MLModel):
            raise ValueError("Only objects of type MLModel can be wrapped with MLModelDecorator instances.")

        self.__dict__["_model"] = model
        return self

    def __getattr__(self, name: str) -> Any:
        """Get an attribute."""
        if name in MLModelDecorator._decorator_attributes:
            return self.__dict__[name]
        else:
            return getattr(self.__dict__["_model"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute."""
        if name in MLModelDecorator._decorator_attributes:
            setattr(self, name, value)
        else:
            setattr(self.__dict__["_model"], name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute."""
        delattr(self.__dict__["_model"], name)

    @property
    def display_name(self) -> str:
        """Property that returns a display name for the model.

        !!! note
            Unless this method is overridden, the implementation just returns the display_name property of the
            model that is being decorated.

        """
        return getattr(self, "_model").display_name

    @property
    def qualified_name(self) -> str:
        """Property that returns the qualified name of the model.

        !!! note
            Unless this method is overridden, the implementation just returns the qualified_name property of the
            model that is being decorated.

        """
        return getattr(self, "_model").qualified_name

    @property
    def description(self) -> str:
        """Property that returns a description of the model.

        !!! note
            Unless this method is overridden, the implementation just returns the description property of the
            model that is being decorated.

        """
        return getattr(self, "_model").description

    @property
    def version(self) -> str:
        """Property that returns the model's version as a string.

        !!! note
            Unless this method is overridden, the implementation just returns the version property of the
            model that is being decorated.


        """
        return getattr(self, "_model").version

    @property
    def input_schema(self) -> BaseModel:
        """Property that returns the schema that is accepted by the predict() method.

        !!! note
            Unless this method is overridden, the implementation just returns the input_schema property of the
            model that is being decorated.

        """
        return getattr(self, "_model").input_schema

    @property
    def output_schema(self) -> BaseModel:
        """Property returns the schema that is returned by the predict() method.

        !!! note
            Unless this method is overridden, the implementation just returns the output_schema property of the
            model that is being decorated.

        """
        return getattr(self, "_model").output_schema

    def predict(self, data: BaseModel) -> BaseModel:
        """Predict with the model.

        Params:
            data: Data used by the model for making a prediction.

        Returns:
            python object -- can be any python type

        !!! note
            Unless this method is overridden, the implementation just calls the predict method of the
            model that is being decorated and returns the result.

        """
        return getattr(self, "_model").predict(data=data)
