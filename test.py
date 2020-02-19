import unittest

import pre_processing as pp
import config


class TestPreProcessing(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pp.load_data(config.bank_additional_train_path)

    def test_validation_catches_missing_nonexistent_poutcome(self):
        self.df.mappings = {'poutcome': {'failure': 0,
                                         'success': 1}}
        self.assertRaises(ValueError, self.df.process_all)


if __name__ == '__main__':
    unittest.main()
