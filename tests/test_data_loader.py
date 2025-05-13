import unittest
import pandas as pd
from src.data_loader import load_data
import os

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'order_id': ['abc123', 'def456'],
            'order_item_id': [1, 2],
            'product_id': ['prod1', 'prod2'],
            'seller_id': ['seller1', 'seller2'],
            'price': [100.0, 200.0],
            'freight_value': [10.0, 20.],
            'customer_id': ['customer1', 'customer2']
        })
        self.test_csv_path = 'temp_test_orders.csv'
        self.test_data.to_csv(self.test_csv_path, index=False)


    def tearDown(self):
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)


    def test_load_data_success(self):
        df = load_data(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 7))
        self.assertListEqual(list(df.columns), [
            'order_id', 'order_item_id', 'product_id',
            'seller_id', 'price', 'freight_value', 'customer_id'
        ])
        self.assertEqual(df['order_item_id'].tolist(), [1, 2])


    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')


if __name__ == '__main__':
    unittest.main()