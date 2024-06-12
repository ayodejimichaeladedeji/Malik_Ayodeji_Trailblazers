import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.email_classifier import email_classifier

class unit_test_translate(unittest.TestCase):

    def setUp(self):
        self.email_classifier = email_classifier("AppGallery.csv")

    def test_preprocess_data(self):
        self.assertEqual(self.email_classifier.proprocess_dataset(), )

if __name__ == '__main__':
    unittest.main()
    