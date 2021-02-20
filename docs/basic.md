# Examples

This example can be executed directly by loading the notebook file in examples/basic.ipynb.


```python
from IPython.display import clear_output
```

To get started we'll install the ml_base package:


```python
!pip install ml_base
clear_output()
```

## Creating a Simple Model

To show how to work with the MLModel base class we'll create a simple model that we can make predictions with. We'll use the scikit-learn library, so we'll need to install it:


```python
!pip install scikit-learn
clear_output()
```

Now we can write some code to train a model:


```python
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
```

## Creating a Wrapper Class for Your Model

Now that we have a model object, we'll define a class that implements the prediction functionality for the code:


```python
import os
from numpy import array


class IrisModel(object):
    def __init__(self):
        dir_path = os.path.abspath('')
        file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, data: dict):
        X = array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)
        y_hat = int(self._svm_model.predict(X)[0])
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        return {"species": species}
```

The class above wraps the pickled model object and makes the model easier to use by converting the inputs and outputs.
To use the model, all we need to do is this:


```python
model = IrisModel()

prediction = model.predict(data={
    "sepal_length":1.0,
    "sepal_width":1.1,
    "petal_length": 1.2,
    "petal_width": 1.3})

prediction
```




    {'species': 'virginica'}



## Creating an MLModel Class for Your Model

The model is already much easier to use because it provides the prediction from a class. The user of the model doesn't
need to worry about loading the pickled model object, or converting the model's input into a numpy array. However, we
are still not using the MLModel abstract base class, now we'll implement a part of the MLModel's interface to show how
it works:


```python
from ml_base import MLModel


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
        raise NotImplementedError()

    @property
    def output_schema(self):
        raise NotImplementedError()

    def __init__(self):
        dir_path = os.path.abspath('')
        file = open(os.path.join(dir_path, "svc_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, data: dict):
        X = array([data["sepal_length"], data["sepal_width"], data["petal_length"], data["petal_width"]]).reshape(1, -1)
        y_hat = int(self._svm_model.predict(X)[0])
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        return {"species": species}
```

The MLModel base class defines a set of properties that must be provided by any class that inherits from it. Because the IrisModel class now provides this metadata about the model, we can access it directly from the model object like this:


```python
model = IrisModel()

print(model.display_name)
```

    Iris Model


The model version is also available as a 


```python
print(model.version)
```

    1.0.0


As you can see, we didn't implement the input_schema and output_schema properties above, we'll add those next.

## Adding Schemas to Your Model

To add schema information to the model class, we'll use the pydantic package:


```python
from pydantic import BaseModel, Field
from pydantic import ValidationError
from enum import Enum


class ModelInput(BaseModel):
    sepal_length: float = Field(gt=5.0, lt=8.0, description="The length of the sepal of the flower.")
    sepal_width: float = Field(gt=2.0, lt=6.0, description="The width of the sepal of the flower.")
    petal_length: float = Field(gt=1.0, lt=6.8, description="The length of the petal of the flower.")
    petal_width: float = Field(gt=0.0, lt=3.0, description="The width of the petal of the flower.")


class Species(str, Enum):
    iris_setosa = "Iris setosa"
    iris_versicolor = "Iris versicolor"
    iris_virginica = "Iris virginica"


class ModelOutput(BaseModel):
    species: Species = Field(description="The predicted species of the flower.")

```

The ModelInput class inherits from the pydantic BaseModel class and it defines four required fields, all of them floating point numbers. The pydantic package allows for defining upper bounds and lower bounds for the values accepted by each field, and also a description for the field.

The ModelOutput is made up of a single fields, which is an enumerated string that contains the predicted species of the flower.

Now that we have the ModelInput and ModelOutput schemas defined as pydantic BaseModel classes, we'll add them to the IrisModel class by returning them from the input_schema and output_schema properties:


```python
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
        dir_path = os.path.abspath('')
        with open(os.path.join(dir_path, "svc_model.pickle"), 'rb') as f:
            self._svm_model = pickle.load(f)

    def predict(self, data: dict):
        model_input = ModelInput(**data)
        
        # creating a numpy array using the fields in the input object
        X = array([model_input.sepal_length, 
                   model_input.sepal_width, 
                   model_input.petal_length, 
                   model_input.petal_width]).reshape(1, -1)
        
        # making a prediction, at this point its a number
        y_hat = int(self._svm_model.predict(X)[0])
        
        # converting the prediction from a number to a string
        targets = ["Iris setosa", "Iris versicolor", "Iris virginica"]
        species = targets[y_hat]
        
        # returning the prediction inside an object
        return ModelOutput(species=species)
```

Notice that we are also using the pydantic models to validate the input before prediction and to
create an object that will be returned from the model's predict() method.

If we use the model class now, we'll get this result:


```python
model = IrisModel()

prediction = model.predict(data={"sepal_length": 6.0, "sepal_width": 2.1, 
                                 "petal_length": 1.2, "petal_width": 1.3})

prediction
```




    ModelOutput(species=<Species.iris_virginica: 'Iris virginica'>)



By adding input and output schemas to the model, we can automate many other operations later. Also, we can query
the model object itself for the schema. The pydantic package is able to create JSON schema from the fields in the input and output schema objects of the model:


```python
model = IrisModel()

model.input_schema.schema()
```




    {'title': 'ModelInput',
     'type': 'object',
     'properties': {'sepal_length': {'title': 'Sepal Length',
       'description': 'The length of the sepal of the flower.',
       'exclusiveMinimum': 5.0,
       'exclusiveMaximum': 8.0,
       'type': 'number'},
      'sepal_width': {'title': 'Sepal Width',
       'description': 'The width of the sepal of the flower.',
       'exclusiveMinimum': 2.0,
       'exclusiveMaximum': 6.0,
       'type': 'number'},
      'petal_length': {'title': 'Petal Length',
       'description': 'The length of the petal of the flower.',
       'exclusiveMinimum': 1.0,
       'exclusiveMaximum': 6.8,
       'type': 'number'},
      'petal_width': {'title': 'Petal Width',
       'description': 'The width of the petal of the flower.',
       'exclusiveMinimum': 0.0,
       'exclusiveMaximum': 3.0,
       'type': 'number'}},
     'required': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']}




```python
model.output_schema.schema()
```




    {'title': 'ModelOutput',
     'type': 'object',
     'properties': {'species': {'$ref': '#/definitions/Species'}},
     'required': ['species'],
     'definitions': {'Species': {'title': 'Species',
       'description': 'An enumeration.',
       'enum': ['Iris setosa', 'Iris versicolor', 'Iris virginica'],
       'type': 'string'}}}



Although it is not required to use the pydantic package to create model schemas, it is recommended. The pydantic
package is installed as a dependency of the ml_base package.

## Using the ModelManager Class

The ModelManager class is provided to help manage model objects. It is a singleton class that is designed to enable
model instances to be instantiated once during the lifecycle of a process and accessed many times:


```python
from ml_base.utilities import ModelManager


model_manager = ModelManager()
```

Because it is a singleton object, a reference to the same object is returned no matter how many times we instantiate it:


```python
print(id(model_manager))

another_model_manager = ModelManager()

print(id(another_model_manager))
```

    5054595472
    5054595472


You can add model instances to the ModelManager singleton by asking it to instantiate the model class:


```python
model_manager.load_model("__main__.IrisModel")
```

The load_model() method is able to find the MLModel class that we defined above and instantiate it, after that it stores a reference to the instance internally.

The ModelManager is also able to save references to model instances that were instantiated in some other way by using the add_model() method:


```python
another_iris_model = IrisModel()

try:
    model_manager.add_model(another_iris_model)
except ValueError as e:
    print(e)
```

    A model with the same qualified name is already in the ModelManager singleton.


In this case, the ModelManager did not save the instance of the IrisModel because we already had an instance of the model. The models are uniquely identified by their qualified name properties.

The ModelManager instance can list the models that it contains with the get_models() method, the details of the instance of IrisModel that we just created are returned:


```python
model_manager.get_models()
```




    [{'display_name': 'Iris Model',
      'qualified_name': 'iris_model',
      'description': 'A model to predict the species of a flower based on its measurements.',
      'version': '1.0.0'}]



The ModelManager instance can return the metadata of any of the models. The metadata includes the input and output schemas as well:


```python
model_manager.get_model_metadata("iris_model")
```




    {'display_name': 'Iris Model',
     'qualified_name': 'iris_model',
     'description': 'A model to predict the species of a flower based on its measurements.',
     'version': '1.0.0',
     'input_schema': {'title': 'ModelInput',
      'type': 'object',
      'properties': {'sepal_length': {'title': 'Sepal Length',
        'description': 'The length of the sepal of the flower.',
        'exclusiveMinimum': 5.0,
        'exclusiveMaximum': 8.0,
        'type': 'number'},
       'sepal_width': {'title': 'Sepal Width',
        'description': 'The width of the sepal of the flower.',
        'exclusiveMinimum': 2.0,
        'exclusiveMaximum': 6.0,
        'type': 'number'},
       'petal_length': {'title': 'Petal Length',
        'description': 'The length of the petal of the flower.',
        'exclusiveMinimum': 1.0,
        'exclusiveMaximum': 6.8,
        'type': 'number'},
       'petal_width': {'title': 'Petal Width',
        'description': 'The width of the petal of the flower.',
        'exclusiveMinimum': 0.0,
        'exclusiveMaximum': 3.0,
        'type': 'number'}},
      'required': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']},
     'output_schema': {'title': 'ModelOutput',
      'type': 'object',
      'properties': {'species': {'$ref': '#/definitions/Species'}},
      'required': ['species'],
      'definitions': {'Species': {'title': 'Species',
        'description': 'An enumeration.',
        'enum': ['Iris setosa', 'Iris versicolor', 'Iris virginica'],
        'type': 'string'}}}}



The ModelManager can return a reference to the instance of any model that it is holding:


```python
iris_model = model_manager.get_model("iris_model")

print(iris_model.display_name)
```

    Iris Model


The instance is identified by the qualified name of the model.

Lastly, a model instance can be removed by calling the remove_model() method:


```python
model_manager.remove_model("iris_model")

model_manager.get_models()
```




    []



To clear the ModelManager instance, you can call the clear_instance() method:


```python
model_manager.clear_instance()
```

To create a new singleton you have to instantiate the ModelManager again:


```python
model_manager = ModelManager()
```
