import unittest
import pandas as pd
from src.models.train import train_model

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'tumor_size': [1.2, 2.3, 3.1, 4.5, 2.2],
            'age': [25, 30, 45, 50, 35],
            'likelihood_of_cancer': [0.1, 0.4, 0.6, 0.9, 0.5]
        })

    def test_train_model(self):
        # Test the model training function
        model = train_model(self.data)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_output(self):
        # Test the output of the model
        model = train_model(self.data)
        predictions = model.predict(self.data[['tumor_size', 'age']])
        self.assertEqual(len(predictions), len(self.data))

if __name__ == '__main__':
    unittest.main()