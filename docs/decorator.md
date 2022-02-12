# MLModelDecorator Example


```python
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("./", os.pardir)))
```

## Creating a Decorator

Decorators are objects that allow us to extend the functionality of other objects at runtime without having to modify the objects that are being decorated. The decorator pattern is a well-known object-oriented design pattern that helps to make code more flexible and reusable.

Notice that we are not working with Python decorators, which are used to decorate functions and methods at loading time only (when the function or class is created). The decorators we will work with are run-time decorators since they are applied during the runtime of the program.

The objects we want to decorate are MLModel objects, so we'll need an MLModel class to work with. We'll create a simple mocked model class to work with along with the input and output schemas:


```python
from ml_base.ml_model import MLModel
from pydantic import BaseModel, Field
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


class IrisModelMock(MLModel):
    display_name = "Iris Model"
    qualified_name = "iris_model"
    description = "A model to predict the species of a flower based on its measurements."
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self):
        pass

    def predict(self, data: ModelInput) -> ModelOutput:
        return ModelOutput(species="Iris setosa")
```

This class mocks the input and output of the IrisModel we used in the previous example. The mocked model will always return a prediction of "Iris setosa". We'll instantiate it to make sure that everything works:


```python
model = IrisModelMock()

prediction = model.predict(
    ModelInput(sepal_length=5.1,
               sepal_width=2.2,
               petal_length=1.2,
               petal_width=1.3))

prediction
```




    ModelOutput(species=<Species.iris_setosa: 'Iris setosa'>)



## Creating a Simple Decorator Class

To create a decorator for MLModel classes, we'll inherit from the MLModelDecorator class:


```python
from ml_base import MLModelDecorator
from ml_base.ml_model import MLModelException


class SimpleDecorator(MLModelDecorator):
    pass
```

The decorator doesn't do anything but it's still useful because it inherits default behavior from the base class. In order to wrap the model instance with a decorator instance, we instantiate the decorator like this:


```python
decorator = SimpleDecorator(model)
```

Now we can make a prediction with the model just like we normally would:


```python
prediction = decorator.predict(
    ModelInput(sepal_length=5.1,
               sepal_width=2.2,
               petal_length=1.2,
               petal_width=1.3))

print(prediction)
```

    species=<Species.iris_setosa: 'Iris setosa'>


The decorator's default implementation of the predict method does nothing but call the corresponding method in the model instance. The same is true for the other parts of the MLModel API.


```python
print(decorator.display_name)
print(decorator.qualified_name)
print(decorator.description)
print(decorator.version)
print(decorator.input_schema)
print(decorator.output_schema)
```

    Iris Model
    iris_model
    A model to predict the species of a flower based on its measurements.
    1.0.0
    <class '__main__.ModelInput'>
    <class '__main__.ModelOutput'>


## Creating an MLModelDecorator With Behavior

The example above wasn't very useful because it didn't do anything. We'll override the default implementation of the MLModelDecorator base class in order to add some behavior.

This decorator executes around the predict() method:


```python
class SimplePredictDecorator(MLModelDecorator):

    def predict(self, data):
        print("Executing before prediction.")
        prediction = self._model.predict(data=data)
        print("Executing after prediction.")
        return prediction
```

The decorator wraps around the predict() method and does nothing except print a message before and after executing the predict method of the model.

We can try it out by wrapping the model instance again:


```python
decorator = SimplePredictDecorator(model)
```

Now, we'll call the predict method:


```python
prediction = decorator.predict(ModelInput(
    sepal_length=5.1,
    sepal_width=2.1,
    petal_length=1.2,
    petal_width=1.3))

prediction
```

    Executing before prediction.
    Executing after prediction.





    ModelOutput(species=<Species.iris_setosa: 'Iris setosa'>)



The decorator instance executed before and after the model's predict() method and printed some messages.

## A More Complex Decorator

The MLModelDecorator class is able to "wrap" every method and property in the MLModel base class. We'll build a more complex MLModelDecorator to show how this works:


```python
class ComplexDecorator(MLModelDecorator):
    
    @property
    def display_name(self) -> str:
        return self._model.display_name + " extra"
    
    @property
    def qualified_name(self) -> str:
        return self._model.qualified_name + " extra"
    
    @property
    def description(self) -> str:
        return self._model.description + " extra"
    
    @property
    def version(self) -> str:
        return self._model.version + " extra"
    
    def predict(self, data):
        print("Executing before prediction.")
        prediction = self._model.predict(data=data)
        print("Executing after prediction.")
        return prediction
```


```python
complex_decorator = ComplexDecorator(model)

print(complex_decorator.display_name)
print(complex_decorator.qualified_name)
print(complex_decorator.description)
print(complex_decorator.version)
```

    Iris Model extra
    iris_model extra
    A model to predict the species of a flower based on its measurements. extra
    1.0.0 extra


The properties of the MLModel instance were modifyied by adding the word "extra" to them, including the input and output schemas, although it would not be a good idea to convert the schema classes to strings in a normal situation.

Any other methods, attributes, or properties of an MLModel class that are not part of the MLModel interface are not modified by MLModelDecorator instances that are wrapping them. To show this we'll create an MLModel class with some extra attributes:


```python
class IrisModelMockWithExtraAttributes(MLModel):
    display_name = "Iris Model"
    qualified_name = "iris_model"
    description = "A model to predict the species of a flower based on its measurements."
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput
    
    def __init__(self):
        self.extra_attribute = "extra_attribute"
    
    def predict(self, data: ModelInput) -> ModelOutput:
        return ModelOutput(species="Iris setosa")
    
    @property
    def extra_property(self):
        return "extra_property"
    
    def extra_method(self):
        return "extra_method"
```


```python
model = IrisModelMockWithExtraAttributes()

decorator = ComplexDecorator(model)

print(decorator.extra_attribute)
print(decorator.extra_property)
print(decorator.extra_method())
```

    extra_attribute
    extra_property
    extra_method


The MLModelDecorator class is designed to execute around the public API of the MLModel base class and stay out of the way of any other part of an MLModel instance.

When implementing decorators, its important to remember to call the method or return the property of the model instance itself, otherwise the decorator would no longer decorate the model, it would just replace it.

## Setting the Model After Initialization

The MLModelDecorator can also be instantiated without a reference to an MLModel instance to decorate.


```python
decorator = ComplexDecorator()

decorator
```




    ComplexDecorator(None)



When we print the decorator, whe model reference inside shows up as "None".

If we try to execute access the API of the decorator, we'll get an error:


```python
try:
    decorator.version
except Exception as e:
    print(e)
```

    'NoneType' object has no attribute 'version'


To set the model instances after initialization, we can use the set_model() method.


```python
decorator.set_model(model)

decorator
```




    ComplexDecorator(IrisModelMockWithExtraAttributes)



Accessing the decorator now accesses the model as show above:


```python
decorator.version
```




    '1.0.0 extra'



## Displaying the Decorator

Once a model instance has been decorated, we can see that it is decorating when we print it:


```python
decorator
```




    ComplexDecorator(IrisModelMockWithExtraAttributes)



The ComplexDecorator is wrapping an instance of MLModelMock.

If we add another decorator, we can see it is decorated again:


```python
decorator = SimplePredictDecorator(decorator)

decorator
```




    SimplePredictDecorator(ComplexDecorator(IrisModelMockWithExtraAttributes))



Decorators can decorate other instances of decorators because they have the same API as MLModel.

## Creating an Exception Handler Decorator

To show a real example of what a decorator can do, we'll create a decorator that handles exceptions raised in the predict() method and logs them.


```python
import logging


logger = logging.getLogger(__name__)


class ExceptionLoggerDecorator(MLModelDecorator):
    
    def predict(self, data):
        try:
            return self._model.predict(data=data)
        except Exception as e:
            logger.exception("Exception in the predict() method of {}.".format(str(self._model)))

```

We'll need to raise an exception in the model class' predict() method in order to try this out, so we'll redefine the IrisModelMock class to raise an exception:


```python
class IrisModelMock(MLModel):
    display_name = "Iris Model"
    qualified_name = "iris_model"
    description = "A model to predict the species of a flower based on its measurements."
    version = "1.0.0"
    input_schema = ModelInput
    output_schema = ModelOutput

    def __init__(self):
        pass
    
    def predict(self, data):
        raise Exception("Exception!")
```

Now all we need is to instantiate the MLModel class and the decorator to try it out:


```python
model = IrisModelMock()

decorator = ExceptionLoggerDecorator(model)

# making a failing prediction
prediction = decorator.predict(ModelInput(
    sepal_length=5.1,
    sepal_width=2.1,
    petal_length=1.2,
    petal_width=1.3))
```

    Exception in the predict() method of IrisModelMock.
    Traceback (most recent call last):
      File "<ipython-input-21-ea4dab4c8b70>", line 11, in predict
        return self._model.predict(data=data)
      File "<ipython-input-22-79739fa901dd>", line 13, in predict
        raise Exception("Exception!")
    Exception: Exception!


The exception was caught by the decorator and logged.

## Configurable MLModel Decorator

Next, we'll build an MLModelDecorator that can be configured.


```python
class AddStringDecorator(MLModelDecorator):
    
    def __init__(self, model: MLModel, extra_name: str) -> None:
        super().__init__(model, extra_name=extra_name)

    @property
    def display_name(self) -> str:
        return self._model.display_name + self._configuration["extra_name"]
```

The \_\_init\_\_() method receives the normal "model" parameter and passes it to the super class. It also receives a parameter called "extra_name" which is also passed to the super class as a keyword argument. Each configuration items should be passed to the super class in this way.



The decorator adds a string to the display_name property of the model object:


```python
model = IrisModelMock()

decorator = AddStringDecorator(model, extra_name=" extra name")
```

Now when we access the properties, we'll get the string we configured added to the end:


```python
print(decorator.display_name)
```

    Iris Model extra name


Once the configuration has been passed to the MLModelDecorator super class as a keyword argument, it is saved in the "\_configuration" attribute and can be accessed by the methods in the decorator class.

This also means that the "\_configuration" and "\_model" names are reserved within MLModelDecorator classes because they are being used by the base class.

You can also set the values in the "\_configuration" and "\_model" attributes of the decorator:


```python
decorator._configuration["asdf"] = "asdf"

decorator._configuration
```




    {'extra_name': ' extra name', 'asdf': 'asdf'}



## Adding the Decorated Model to the ModelManager

Adding a decorated model to the ModelManager singleton is simple. First we'll create a decorated model:


```python
model = IrisModelMock()

decorated_model = SimpleDecorator(model)
```

Next, we'll create the ModelManager:


```python
from ml_base.utilities import ModelManager


model_manager = ModelManager()
```

Finally, we'll add the decorated model as we normally would:


```python
model_manager.add_model(decorated_model)

model_manager.get_model_metadata("iris_model")
```




    {'display_name': 'Iris Model',
     'qualified_name': 'iris_model',
     'description': 'A model to predict the species of a flower based on its measurements.',
     'version': '1.0.0',
     'input_schema': {'title': 'ModelInput',
      'type': 'object',
      'properties': {'sepal_length': {'title': 'Sepal Length',
        'exclusiveMinimum': 5.0,
        'exclusiveMaximum': 8.0,
        'type': 'number'},
       'sepal_width': {'title': 'Sepal Width',
        'exclusiveMinimum': 2.0,
        'exclusiveMaximum': 6.0,
        'type': 'number'},
       'petal_length': {'title': 'Petal Length',
        'exclusiveMinimum': 1.0,
        'exclusiveMaximum': 6.8,
        'type': 'number'},
       'petal_width': {'title': 'Petal Width',
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



The ModelManager is able to work with the decorated model object because it has the same interface as MLModel.


```python
model_manager.clear_instance()
```

## Adding a Decorator to a Model in the ModelManager

The ModelManager also has support for decorating models that are already held inside by using the add_decorator() method:


```python
from ml_base.utilities import ModelManager

model_manager = ModelManager()

model = IrisModelMock()

model_manager.add_model(model)

print(model_manager.get_models())
```

    [{'display_name': 'Iris Model', 'qualified_name': 'iris_model', 'description': 'A model to predict the species of a flower based on its measurements.', 'version': '1.0.0'}]



```python
decorator = SimpleDecorator()

model_manager.add_decorator("iris_model", decorator)
```

When we access the model instance, we can see that it is now decorated:


```python
model = model_manager.get_model("iris_model")

model
```




    SimpleDecorator(IrisModelMock)


