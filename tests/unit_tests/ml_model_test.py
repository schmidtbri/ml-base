import unittest
import json

from ml_base.ml_model import MLModelSchemaValidationException
from tests.mocks import IrisModelMock


class MLModelTests(unittest.TestCase):

    def test_get_json_schema(self):
        # arrange
        instance = IrisModelMock()

        # act
        json_schema_string = instance.input_schema.schema_json()
        json_schema = json.loads(json_schema_string)

        # assert
        self.assertTrue(type(json_schema_string) is str)
        self.assertTrue(json_schema["title"] == "ModelInput")
        self.assertTrue(json_schema["type"] == "object")


if __name__ == '__main__':
    unittest.main()
