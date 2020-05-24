from ml_base.ml_model import MLModel


# creating an MLModel class to test with
class MLModelMock(MLModel):
    # accessing the package metadata
    display_name = "display name"
    qualified_name = "qualified_name"
    description = "description"
    version = "1.0.0"
    input_schema = None
    output_schema = None

    def __init__(self):
        pass

    def predict(self, data):
        pass


# creating a mockup class to test with
class SomeClass(object):
    pass
