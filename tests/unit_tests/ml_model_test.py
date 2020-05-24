import unittest
from traceback import print_tb

from tests.mocks import MLModelMock, SomeClass
from ml_base.utilities.model_manager import ModelManager


class MLModelTests(unittest.TestCase):

    def test1(self):
        """Testing the load_models() method."""
        # arrange

        # act

        # assert
        self.assertFalse(False)


if __name__ == '__main__':
    unittest.main()
