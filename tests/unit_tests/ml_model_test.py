import unittest
import json

from ml_base.ml_model import MLModelSchemaValidationException
from tests.mocks import MLModelMock


class MLModelTests(unittest.TestCase):

    def test_throw_exception_on_bad_input(self):
        """Testing that the MLModel class throws exception on bad input."""
        # arrange
        instance = MLModelMock()

        # act
        exception_raised = False
        exception_type = None
        try:
            instance.predict({"sepal_length": 10.0,
                              "sepal_width": 2.5,
                              "petal_length": 1.2,
                              "petal_width": 1.3})
        except Exception as e:
            exception_raised = True
            exception_type = type(e)

        # assert
        print(exception_type == MLModelSchemaValidationException)
        self.assertTrue(exception_raised)

    def test_get_json_schema(self):
        # arrange
        instance = MLModelMock()

        # act
        json_schema_string = instance.input_schema.schema_json()
        json_schema = json.loads(json_schema_string)

        # assert
        self.assertTrue(type(json_schema_string) is str)
        self.assertTrue(json_schema["title"] == "ModelInput")
        self.assertTrue(json_schema["type"] == "object")


if __name__ == '__main__':
    unittest.main()
