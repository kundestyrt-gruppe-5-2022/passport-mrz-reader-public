"""Custom implementation for checking TD3 passports. It checks that the passport
has the correct format and that the checksums are correct.
"""

import re
from typing import TypedDict, Optional
from passport_mrz_reader.common.mrz_common import MRZ_CHARACTERS


class Interval(TypedDict):
    """Stores the start and end of an interval"""

    start: int
    end: int


class CustomPassportChecker:
    """
    Used to check that MRZ-codes are valid
    """

    def __init__(self, line1: str, line2: str, ignore_first_line=True):
        self._line1: str = line1
        self._line2: str = line2
        self._line2_as_digits: list[int] = process_characters(line2)
        self.ignore_first_line: bool = ignore_first_line
        self._reasons: list[str] = []

    def _check_first_line(self) -> bool:
        """
        Used to check wheter the first line confirms to passport standards

        This is the regex used to check: (P[A-Z<][A-Z]{3}([A-Z]{2,}<?)*<<([A-Z]{2,}<?)*<*)
        """
        if len(self._line1) != 44:
            self._reasons.append("First line is not 44 characters")
            return False

        correct = bool(
            re.fullmatch(
                "P[A-Z<][A-Z](([A-Z][A-Z<])|(<<))([A-Z]{2,}<?)*<<([A-Z]{2,}<?)*<*",
                self._line1,
            )
        )

        if not correct:
            self._reasons.append("First line did not pass checksum test")

        return correct

    def _check_second_line(self) -> bool:
        """
        Used to check the format of the second line
        """
        if len(self._line2) != 44:
            self._reasons.append("Second line is not 44 characters")
            return False

        correct = bool(
            re.fullmatch(
                r"[A-Z0-9<]{9}\d"  # Document number + check digit 1
                r"([A-Z]([A-Z][A-Z<]|<<))"  # Nationality
                r"\d{7}"  # Date of birth + check digit 2
                r"[FM<]"  # Gender
                r"\d{7}"  # Expiry date + check digit 3
                r"([A-Z0-9<]{14}\d|<{15})"  # Optional data + check digit 4
                r"\d",  # Master check digit
                self._line2,
            )
        )

        if not correct:
            self._reasons.append("Wrong format for second line")

        return correct

    def _check_second_line_part1(self) -> bool:
        """
        Used to check the first part of the second line is correct
        """
        correct = self.verify(self._line2, [{"start": 0, "end": 9}], 9)
        if not correct:
            self._reasons.append("First checksum failed")
        return correct

    def _check_second_line_part2(self) -> bool:
        """
        Used to check the second part of the second line
        """
        correct = self.verify(self._line2, [{"start": 13, "end": 19}], 19)
        if not correct:
            self._reasons.append("Second checksum failed")
        return correct

    def _check_second_line_part3(self) -> bool:
        """
        Used to check the third part of the second line
        """
        correct = self.verify(self._line2, [{"start": 21, "end": 27}], 27)
        if not correct:
            self._reasons.append("third checksum failed")
        return correct

    def _check_second_line_part4(self) -> bool:
        """
        Used to check the fourth part of the second line
        """
        correct = (re.match("^<+$", self._line2[28:43])) or self.verify(
            self._line2, [{"start": 28, "end": 42}], 42
        )

        if not correct:
            self._reasons.append("Fourth checksum failed")
        return correct

    def _check_master_checksum(self) -> bool:
        """
        Used to check the master checksum
        """
        correct = self.verify(
            self._line2,
            [
                {"start": 0, "end": 10},
                {"start": 13, "end": 20},
                {"start": 21, "end": 43},
            ],
            43,
        )

        if not correct:
            self._reasons.append("Master checksum failed")
        return correct

    def is_correct(self) -> bool:
        """
        Checks if passport follows TD3 standard
        """
        self._reasons = []

        one = self._check_first_line() if not self.ignore_first_line else True
        two = self._check_second_line()

        if not two:
            # Make sure no value errors occur in the upcoming checks
            return False

        three = self._check_second_line_part1()
        four = self._check_second_line_part2()
        five = self._check_second_line_part3()
        six = self._check_second_line_part4()
        master = self._check_master_checksum()

        return all([one, two, three, four, five, six, master])

    def get_reasons_failing(self) -> Optional[list[str]]:
        """
        Get the reasons why the passport is invalid, if any
        """
        valid = self.is_correct()
        return self._reasons if not valid else None

    def verify(
        self, input_string: str, intervals: list[Interval], check_index: int
    ) -> bool:
        """Verify the checksum"""
        letters_list = [
            input_string[interval["start"] : interval["end"]] for interval in intervals
        ]
        letters = "".join(letters_list)
        processed_characters = process_characters(letters)
        weighted_values = weigh_values(processed_characters)
        total = sum(weighted_values)
        return total % 10 == int(input_string[check_index])


def letter_to_number(letter: str) -> int:
    """
    Converts a letter to a number for MRZ reading
    """
    if letter not in MRZ_CHARACTERS:
        raise ValueError(
            "Letter not a valid MRZ character. Only A-Z, 0-9 and < allowed"
        )
    if re.match("[A-Z]", letter):
        return ord(letter) - ord("A") + 10
    if letter == "<":
        return 0
    return int(letter)


def process_characters(string: str) -> list[int]:
    """Encode characters to numbers to be used in checksum calculation"""
    return [letter_to_number(letter) for letter in string]


def weigh_values(arr: list[int]) -> list[int]:
    """Weigh values according to MRZ standard"""
    weights = [7, 3, 1]
    return [item * weights[i % 3] for i, item in enumerate(arr)]
