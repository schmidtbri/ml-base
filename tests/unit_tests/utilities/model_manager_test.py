import unittest
from traceback import print_tb

from ml_base.utilities.model_manager import ModelManager
from tests.mocks import SomeClass, IrisModelMock, SimpleDecorator


class ModelManagerTests(unittest.TestCase):

    def test_model_manager_will_return_same_instance_when_instantiated_many_times(self):
        # arrange, act
        # instantiating the model manager class twice
        first_model_manager = ModelManager()
        second_model_manager = ModelManager()

        # loading the MLModel objects from configuration
        first_model_manager.load_model("tests.mocks.IrisModelMock")

        first_model_object = first_model_manager.get_model(qualified_name="qualified_name")
        second_model_object = second_model_manager.get_model(qualified_name="qualified_name")

        # assert
        self.assertTrue(str(first_model_manager) == str(second_model_manager))
        self.assertTrue(str(first_model_object) == str(second_model_object))

        # cleanup
        first_model_manager.clear_instance()

    def test_model_manager_will_not_execute_init_twice(self):
        # arrange
        # instantiating the model manager class twice
        first_model_manager = ModelManager()
        first_model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        second_model_manager = ModelManager()

        # assert
        self.assertTrue(first_model_manager._models != [])

        # cleanup
        first_model_manager.clear_instance()

    def test_load_model_method(self):
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # adding the model
        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        exception_raised = False
        model_object = None
        # accessing the IrisModelMock model object
        try:
            model_object = model_manager.get_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised = True
            print_tb(e)

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(model_object is not None)

        # cleanup
        model_manager.clear_instance()

    def test_load_model_method_with_wrong_class_path(self):
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # act
        # adding the model
        exception_raised = False
        exception_message = None
        # accessing the IrisModelMock model object
        try:
            model_manager.load_model("sdf.sdf.sdf")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "No module named 'sdf'")

        # cleanup
        model_manager.clear_instance()

    def test_only_ml_model_instances_allowed_to_be_stored(self):
        # arrange
        model_manager = ModelManager()
        some_object = SomeClass()

        # act
        exception_raised = False
        exception_message = ""
        try:
            model_manager.add_model(some_object)
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "ModelManager instance can only hold references to objects of type MLModel.")

        # cleanup
        model_manager.clear_instance()

    def test_model_manager_does_not_allow_duplicate_qualified_names(self):
        # arrange
        model_manager = ModelManager()
        model1 = IrisModelMock()
        model2 = IrisModelMock()

        # act
        # loading the first instance of the model object
        model_manager.add_model(model1)

        exception_raised = False
        exception_message = ""
        try:
            # loading it again
            model_manager.add_model(model2)
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "A model with the same qualified name is already in the ModelManager singleton.")

        # cleanup
        model_manager.clear_instance()

    def test_remove_model_method(self):
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # adding the model
        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        exception_raised1 = False
        # accessing the IrisModelMock model object
        try:
            model_manager.remove_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised1 = True

        exception_raised2 = False
        exception_message2 = ""
        # trying to access the model that was removed
        try:
            model = model_manager.get_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised2 = True
            exception_message2 = str(e)

        # assert
        self.assertFalse(exception_raised1)
        self.assertTrue(exception_raised2)
        self.assertTrue(exception_message2 == "Instance of model 'qualified_name' not found in ModelManager.")

        # cleanup
        model_manager.clear_instance()

    def test_remove_model_method_with_missing_model(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        exception_raised = False
        exception_message = ""
        try:
            model_manager.remove_model(qualified_name="asdf")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "Instance of model 'asdf' not found in ModelManager.")

        # cleanup
        model_manager.clear_instance()

    def test_get_models_method(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        models = model_manager.get_models()

        # assert
        self.assertTrue(models[0]["display_name"] == "display_name")
        self.assertTrue(models[0]["qualified_name"] == "qualified_name")
        self.assertTrue(models[0]["description"] == "description")
        self.assertTrue(models[0]["version"] == "1.0.0")

        # cleanup
        model_manager.clear_instance()

    def test_get_model_metadata_method(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        model_metadata = model_manager.get_model_metadata(qualified_name="qualified_name")

        # assert
        self.assertTrue(model_metadata["display_name"] == "display_name")
        self.assertTrue(model_metadata["qualified_name"] == "qualified_name")
        self.assertTrue(model_metadata["description"] == "description")
        self.assertTrue(model_metadata["version"] == "1.0.0")
        self.assertTrue(type(model_metadata["input_schema"]) is dict)
        self.assertTrue(type(model_metadata["output_schema"]) is dict)

        # cleanup
        model_manager.clear_instance()

    def test_get_model_metadata_method_with_missing_model(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        excpeption_raised = False
        exception_message = None
        try:
            model_metadata = model_manager.get_model_metadata(qualified_name="asdf")
        except Exception as e:
            excpeption_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(excpeption_raised)
        self.assertTrue(exception_message == "Instance of model 'asdf' not found in ModelManager.")

        # cleanup
        model_manager.clear_instance()

    def test_get_model_method(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        exception_raised = False
        model = None
        try:
            model = model_manager.get_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(type(model) is IrisModelMock)

        # cleanup
        model_manager.clear_instance()

    def test_get_model_method_with_missing_model(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        # act
        exception_raised = False
        exception_message = ""
        model = None
        try:
            model = model_manager.get_model(qualified_name="asdf")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "Instance of model 'asdf' not found in ModelManager.")

        # cleanup
        model_manager.clear_instance()

    def test_add_decorator_method(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        decorator = SimpleDecorator()

        # act
        model_manager.add_decorator("qualified_name", decorator)

        model = model_manager.get_model("qualified_name")

        # assert
        self.assertTrue(str(model) == "SimpleDecorator(IrisModelMock)")

        # cleanup
        model_manager.clear_instance()

    def test_add_decorator_method_with_missing_model(self):
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.IrisModelMock")

        decorator = SimpleDecorator()

        # act, assert
        with self.assertRaises(ValueError) as context:
            model_manager.add_decorator("asdf", decorator)

        # cleanup
        model_manager.clear_instance()


if __name__ == '__main__':
    unittest.main()
