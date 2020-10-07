*******
Example
*******

Creating a Simple Model
#######################

To show how to work with the MLModel base class we'll create a simple model that we can make predictions with. We'll use
the scikit-learn library::

    pip install scikit-learn

Now we can write some code::

    from sklearn import datasets
    from sklearn import svm
    import pickle

    # loading the Iris dataset
    iris = datasets.load_iris()

    # instantiating an SVM model from scikit-learn
    svm_model = svm.SVC(gamma=1.0, C=1.0)

    # fitting the model
    svm_model.fit(iris.data[:-1], iris.target[:-1])

    # serializing the model and saving it
    file = open("svc_model.pickle", 'wb')
    pickle.dump(svm_model, file)
    file.close()


Creating a Wrapper Class for Your Model
#######################################
Now that we have a model object, we'll define a class that implements the prediction functionality for the code::

    import os
    import pickle
    from numpy import array


    class IrisModel(object):
        def __init__(self):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
            self._svm_model = pickle.load(file)
            file.close()

        def predict(self, data: dict):
            X = array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)
            y_hat = int(self._svm_model.predict(X)[0])
            targets = ['setosa', 'versicolor', 'virginica']
            species = targets[y_hat]
            return {"species": species}


The class above wraps the pickled model object and makes the model easier to use by converting the inputs and outputs.
To use the model, all we need to do is this::

    model = IrisModel()
    prediction = model.predict(data={
        "sepal_length":1.0,
        "sepal_width":1.1,
        "petal_length": 1.2,
        "petal_width": 1.3})


Creating an MLModel Class for Your Model
########################################

The model is already much easier to use because it provides the prediction from a class. The user of the model doesn't
need to worry about loading the pickled model object, or converting the model's input into a numpy array. However, we
are still not using the MLModel abstract base class, now we'll implement a part of the MLModel's interface to show how
it works::

    import os
    import pickle
    from numpy import array
    from ml_base import MLModel

    class c(MLModel):
        @property
        def display_name(self):
            return "Iris Model"

        @property
        def qualified_name(self):
            return "iris_model"

        @property
        def description(self):
            return "A model to predict the species of a flower based on its measurements."

        @property
        def version(self):
            return "1.0.0"

        @property
        def input_schema(self):
            raise NotImplementedError()

        @property
        def output_schema(self):
            raise NotImplementedError()

        def __init__(self):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
            self._svm_model = pickle.load(file)
            file.close()

        def predict(self, data: dict):
            X = array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)
            y_hat = int(self._svm_model.predict(X)[0])
            targets = ['setosa', 'versicolor', 'virginica']
            species = targets[y_hat]
            return {"species": species}

The MLModel base class defines a set of properties that must be provided by any class that inherits from it. Because the
IrisModel class now provides this metadata about the model, we can access it directly from the model object like this::

    >>> model = IrisModel()
    >>> print(model.display_name)
    Iris Model
    >>> print(model.version)
    1.0.0

As you can see, we didn't implement the schema properties above, we'll add those next.

Adding Schema to Your Model
###########################

To add schema information to the model class, we'll use the pydantic package::

    from pydantic import BaseModel, Field
    from pydantic import ValidationError
    from enum import Enum

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

Now that we have the ModelInput and ModelOutput schemas defined as pydantic BaseModel classes, we'll add
them to the IrisModel class, returning them from the input_schema and output_schema properties::

    import os
    import pickle
    from numpy import array
    from ml_base.ml_model import MLModel, MLModelSchemaValidationException

    class IrisModel(MLModel):
        @property
        def display_name(self):
            return "Iris Model"

        @property
        def qualified_name(self):
            return "iris_model"

        @property
        def description(self):
            return "A model to predict the species of a flower based on its measurements."

        @property
        def version(self):
            return "1.0.0"

        @property
        def input_schema(self):
            return ModelInput

        @property
        def output_schema(self):
            return ModelOutput

        def __init__(self):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
            self._svm_model = pickle.load(file)
            file.close()

        def predict(self, data: dict):
            model_input = ModelInput(**data)
            X = array([model_input.sepal_length, model_input.sepal_width, model_input.petal_length, model_input.petal_width]).reshape(1, -1)
            y_hat = int(self._svm_model.predict(X)[0])
            targets = ["Iris setosa", "Iris versicolor", "Iris virginica"]
            species = targets[y_hat]
            return ModelOutput(species=species)

Notice that we are also using the pydantic models to validate the input before prediction and to
create an object that will be returned from the model's predict() method.

If we use the model class now, we'll get this result::

    >>> model = IrisModel()
    >>> prediction = model.predict(data={"sepal_length":6.0, "sepal_width":2.1, "petal_length": 1.2, "petal_width": 1.3})
    >>> print(prediction)
    ModelOutput(species=<Species.iris_virginica: 'Iris virginica'>)


By adding input and output schemas to the model, we can automate many other operations later. Also, we can query
the model object itself for the schema::

    >>> model = IrisModel()
    >>> print(model.input_schema.schema())
    {'title': 'ModelInput', 'type': 'object', 'properties': {'sepal_length': ...
    >>> print(model.output_schema.schema())
    {'title': 'ModelOutput', 'type': 'object', 'properties': {'species': ...

Although it is not required to use the pydantic package to create model schemas, it is recommended. The pydantic
package is installed as a dependency of the ml_base package.

Using the ModelManager Class
############################

The ModelManager class is provided to help manage model objects. It is a singleton class that is designed to enable
model instances to be instantiated once during the lifecycle of a process and accessed many times::

    >>> from ml_base.utilities import ModelManager

    >>> model_manager = ModelManager()
    >>> model_manager.load_model("__main__.IrisModel")

The load_model() method is able to find the MLModel class that we defined above and instantiates it.

The ModelManager instance can list the models being managed::

    >>> model_manager.get_models()
    [{'display_name': 'Iris Model', 'qualified_name': 'iris_model', ...

The ModelManager instance can return the metadata of one of the models::

    >>> model_manager.get_model_metadata("iris_model")
    {'display_name': 'Iris Model', 'qualified_name': 'iris_model', 'description': ...

The ModelManager will return a reference to a model object like this::

    >>> iris_model = model_manager.get_model("iris_model")
    >>> iris_model
    <__main__.IrisModel object at 0x10e658390>