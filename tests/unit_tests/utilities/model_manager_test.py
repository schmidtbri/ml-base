import unittest
from traceback import print_tb

from ml_base.utilities.model_manager import ModelManager
from tests.mocks import MLModelMock


class ModelManagerTests(unittest.TestCase):

    def test_model_manager_will_return_same_instance_when_instantiated_many_times(self):
        """Testing that the ModelManager will return the same instance of an MLModel class from several different
        references of ModelManager."""
        # arrange, act
        # instantiating the model manager class twice
        first_model_manager = ModelManager()
        second_model_manager = ModelManager()

        # loading the MLModel objects from configuration
        first_model_manager.load_model("tests.mocks.MLModelMock")

        first_model_object = first_model_manager.get_model(qualified_name="qualified_name")
        second_model_object = second_model_manager.get_model(qualified_name="qualified_name")

        # assert
        self.assertTrue(str(first_model_manager) == str(second_model_manager))
        self.assertTrue(str(first_model_object) == str(second_model_object))

    def test_load_model_method(self):
        """Testing the load_model() method."""
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # adding the model
        model_manager.load_model("tests.mocks.MLModelMock")

        # act
        exception_raised = False
        model_object = None
        # accessing the MLModelMock model object
        try:
            model_object = model_manager.get_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised = True
            print_tb(e)

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(model_object is not None)

    def test_load_model_method_with_wrong_class_path(self):
        """Testing the load_model() method."""
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # act
        # adding the model
        exception_raised = False
        exception_message = None
        # accessing the MLModelMock model object
        try:
            model_manager.load_model("sdf.sdf.sdf")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "No module named 'sdf'")

    def test_only_ml_model_instances_allowed_to_be_stored(self):
        """Testing that the ModelManager only allows MLModel objects to be stored."""
        # arrange
        model_manager = ModelManager()

        # act
        exception_raised = False
        exception_message = ""
        try:
            model_manager.load_model("tests.mocks.SomeClass")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "ModelManager instance can only hold references to objects of type MLModel.")

    def test_model_manager_does_not_allow_duplicate_qualified_names(self):
        """Testing that the ModelManager does not allow duplicate qualified names in the singleton."""
        # arrange
        model_manager = ModelManager()

        # act
        # loading the first instance of the model object
        model_manager.load_model("tests.mocks.MLModelMock")

        exception_raised = False
        exception_message = ""
        try:
            # loading it again
            model_manager.load_model("tests.mocks.MLModelMock")
        except Exception as e:
            exception_raised = True
            exception_message = str(e)

        # assert
        self.assertTrue(exception_raised)
        self.assertTrue(exception_message == "A model with the same qualified name is already in the ModelManager singleton.")

    def test_remove_model_method(self):
        """Testing the remove_model() method."""
        # arrange
        # instantiating the model manager class
        model_manager = ModelManager()

        # adding the model
        model_manager.load_model("tests.mocks.MLModelMock")

        # act
        exception_raised1 = False
        # accessing the MLModelMock model object
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

    def test_remove_model_method_with_missing_model(self):
        """Testing that the ModelManager raises ValueError exception when removing a model that is not found."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

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

    def test_get_models_method(self):
        """Testing get_models method."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

        # act
        models = model_manager.get_models()

        # assert
        self.assertTrue(models[0]["display_name"] == "display_name")
        self.assertTrue(models[0]["qualified_name"] == "qualified_name")
        self.assertTrue(models[0]["description"] == "description")
        self.assertTrue(models[0]["version"] == "1.0.0")

    def test_get_model_metadata_method(self):
        """Testing get_model_metadata method."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

        # act
        model_metadata = model_manager.get_model_metadata(qualified_name="qualified_name")

        # assert
        self.assertTrue(model_metadata["display_name"] == "display_name")
        self.assertTrue(model_metadata["qualified_name"] == "qualified_name")
        self.assertTrue(model_metadata["description"] == "description")
        self.assertTrue(model_metadata["version"] == "1.0.0")
        self.assertTrue(type(model_metadata["input_schema"]) is dict)
        self.assertTrue(type(model_metadata["output_schema"]) is dict)

    def test_get_model_metadata_method_with_missing_model(self):
        """Testing get_model_metadata method with missing model."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

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

    def test_get_model_method(self):
        """Testing the get_model method."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

        # act
        exception_raised = False
        model = None
        try:
            model = model_manager.get_model(qualified_name="qualified_name")
        except Exception as e:
            exception_raised = True

        # assert
        self.assertFalse(exception_raised)
        self.assertTrue(type(model) is MLModelMock)

    def test_get_model_method_with_missing_model(self):
        """Testing that the ModelManager raises ValueError exception when a model is not found."""
        # arrange
        model_manager = ModelManager()

        model_manager.load_model("tests.mocks.MLModelMock")

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


if __name__ == '__main__':
    unittest.main()
