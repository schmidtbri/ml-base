import unittest

from ml_base.ml_model import MLModelException, MLModel
from tests.mocks import SomeClass, IrisModelMock, IrisModelMockThatRaisesException, SimpleDecorator,\
    ModelInput, ModelOutput, AddStringDecorator, CatchExceptionsDecorator, ModifySchemaDecorator


class DecoratorTests(unittest.TestCase):

    def test_str_method(self):
        """Test that the __str__ method of a decorator works."""
        # arrange
        model = IrisModelMock()
        decorator1 = SimpleDecorator(model=model)
        decorator2 = AddStringDecorator(model=decorator1, string=" test")

        # act
        string = str(decorator2)

        # assert
        self.assertTrue(string == "AddStringDecorator(SimpleDecorator(IrisModelMock))")

    def test_dynamic_attribute_access(self):
        """Test that all attributes of MLModel object can be accessed through decorators."""
        # arrange
        model = IrisModelMock()
        decorator = SimpleDecorator(model=model)

        # act, assert
        decorator.new_attribute = "asd"
        self.assertTrue(model.new_attribute == "asd")

        # act, assert
        del(decorator.new_attribute)
        self.assertTrue("new_attribute" not in model.__dict__)

        # now doing the same thing with more than one decorator
        another_decorator = AddStringDecorator(model=decorator, string=" test")

        # act, assert
        another_decorator.new_attribute = "asd"
        self.assertTrue(model.new_attribute == "asd")

        # act, assert
        del(another_decorator.new_attribute)
        self.assertTrue("new_attribute" not in model.__dict__)

    def test_decorating_an_object_that_is_not_of_type_ml_model(self):
        """Testing decorating an object that is not of type MLModel."""
        # arrange
        model = SomeClass()

        # act, assert
        with self.assertRaises(ValueError) as context:
            decorator = SimpleDecorator(model=model)

    def test_with_simple_decorator(self):
        """Testing that the SimpleDecorator class works."""
        # arrange
        model = IrisModelMock()
        decorator = SimpleDecorator(model=model)

        # act
        display_name = decorator.display_name
        qualified_name = decorator.qualified_name
        description = decorator.description
        version = decorator.version
        input_schema = decorator.input_schema
        output_schema = decorator.output_schema
        prediction = decorator.predict(data={"sepal_length": 6.0,
                                             "sepal_width": 4.0,
                                             "petal_length": 2.0,
                                             "petal_width": 1.0})

        # assert
        self.assertTrue(display_name == "display_name")
        self.assertTrue(qualified_name == "qualified_name")
        self.assertTrue(description == "description")
        self.assertTrue(version == "1.0.0")
        self.assertTrue(input_schema == ModelInput)
        self.assertTrue(output_schema == ModelOutput)
        self.assertTrue(type(prediction) is ModelOutput)

    def test_with_add_string_decorator(self):
        """Testing that the AddStringDecorator class works."""
        # arrange
        model = IrisModelMock()
        decorator = AddStringDecorator(model=model, string=" test")

        # act
        display_name = decorator.display_name
        qualified_name = decorator.qualified_name
        description = decorator.description
        version = decorator.version
        input_schema = decorator.input_schema
        output_schema = decorator.output_schema
        prediction = decorator.predict(data={"sepal_length": 6.0,
                                             "sepal_width": 4.0,
                                             "petal_length": 2.0,
                                             "petal_width": 1.0})

        # assert
        self.assertTrue(display_name == "display_name test")
        self.assertTrue(qualified_name == "qualified_name test")
        self.assertTrue(description == "description test")
        self.assertTrue(version == "1.0.0 test")
        self.assertTrue(input_schema == ModelInput)
        self.assertTrue(output_schema == ModelOutput)
        self.assertTrue(type(prediction) is ModelOutput)

    def test_with_catch_exception_decorator(self):
        """Testing that the CatchExceptionsDecorator class works."""
        # arrange
        model = IrisModelMockThatRaisesException()
        decorator = CatchExceptionsDecorator(model=model)

        # act
        with self.assertRaises(MLModelException) as context:
            prediction = decorator.predict(data={"sepal_length": 6.0,
                                                 "sepal_width": 4.0,
                                                 "petal_length": 2.0,
                                                 "petal_width": "asdf"})

        # assert
        self.assertTrue(decorator.display_name == "display_name")
        self.assertTrue(decorator.qualified_name == "qualified_name")
        self.assertTrue(decorator.description == "description")
        self.assertTrue(decorator.version == "1.0.0")
        self.assertTrue(decorator.input_schema == ModelInput)
        self.assertTrue(decorator.output_schema == ModelOutput)

    def test_with_modify_schema_decorator(self):
        """Testing that modifying the schema of a model is possible in a decorator."""
        # arrange
        model = IrisModelMock()
        decorator = ModifySchemaDecorator(model=model)

        # act, assert
        data = {"sepal_length": 6.0,
                "sepal_width": 4.0,
                "petal_length": 2.0,
                "petal_width": 1.5,
                "correlation_id": "7c687688-cc7e-4d96-b9db-3ae10d5bd19a"}

        p = decorator.input_schema(**data)

    def test_is_instance_check_works_correctly(self):
        """Testing that a decorator will pass the isinstance check as a subtype of MLModel."""
        # arrange
        model = IrisModelMock()
        decorator = ModifySchemaDecorator(model=model)

        # act, assert
        self.assertTrue(isinstance(decorator, MLModel))


if __name__ == '__main__':
    unittest.main()
