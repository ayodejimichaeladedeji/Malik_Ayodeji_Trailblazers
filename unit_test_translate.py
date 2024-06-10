import unittest
from email_classifier import email_classifier

class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.email_classifier = email_classifier()

    def test_translate_method(self):
        self.assertEqual(self.email_classifier
                        .trans_to_en(["Wichtige E-Mail", "Fehlgeschlagen", "Unterbereitstellung", "Cloud", "Festplatte"]), 
                        ["important email", "failed", "under-provisioning", "cloud", "hard disk drive"])

if __name__ == '__main__':
    unittest.main()
    