"""Tests the custom passport checker"""

import unittest
from passport_mrz_reader.utils.custom_passport_checker import (
    CustomPassportChecker,
)


class TestCustomPassportChecker(unittest.TestCase):
    """Tests the custom passport checker"""

    def test_valid_passport(self):
        """Test a valid passport"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertTrue(valid)

    def test_german_passport(self):
        """Test a valid German passport with different country code format"""
        line1 = "P<D<<GOMEZ<<HENRICH<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertTrue(valid)

    def test_invalid_passport(self):
        """Test an invalid passport"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7406122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertFalse(valid)

    def test_number_on_first_line(self):
        """Test a passport with a number on the first line"""
        line1 = "P<UTOERIKSSON<<ANNA<MAR1A<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2, ignore_first_line=False).is_correct()
        self.assertFalse(valid)

    def test_number_on_first_line_ignored(self):
        """Test a passport with a number on the first line, but first line ignored"""
        line1 = "P<UTOERIKSSON<<ANNA<MAR1A<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2, ignore_first_line=True).is_correct()
        self.assertTrue(valid)

    def test_wrong_document_type(self):
        """Test a passport with a wrong document type"""
        line1 = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2, ignore_first_line=False).is_correct()
        self.assertFalse(valid)

    def test_wrong_document_type_ignored(self):
        """Test a passport with a wrong document type, but first line ignored"""
        line1 = "V<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2, ignore_first_line=True).is_correct()
        self.assertTrue(valid)

    def test_number_on_nationality_field(self):
        """Tests a passport with a digit on the nationality field"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UT07408122F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertFalse(valid)

    def test_letter_on_date(self):
        """Tests a passport where the date of birth has a letter"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408I22F1204159ZE184226B<<<<<10"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertFalse(valid)

    def test_letter_on_checksum(self):
        """Tests a passport where the checksum has a letter"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<1O"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertFalse(valid)

    def test_letter_wrong_line_length(self):
        """Tests a passport where the line length is wrong"""
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<1"
        valid = CustomPassportChecker(line1, line2).is_correct()
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
