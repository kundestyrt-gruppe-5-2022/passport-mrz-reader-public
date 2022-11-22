"""Contains common constants and helper functions to be used
by the other scripts in this directory.
"""
from IPython.display import display

# Constants

PASSPORT_FIELDS = {
    "document_type": (0, 0, 2),
    "country_code": (0, 2, 5),
    "full name": (0, 5, 44),
    "document_number_hash": (1, 9, 10),
    "nationality": (1, 10, 13),
    "birth_date": (1, 13, 19),
    "birth_date_hash": (1, 19, 20),
    "expiry_date": (1, 21, 27),
    "expiry_date_hash": (1, 27, 28),
    "optional_data_hash": (1, 42, 43),
    "final_hash": (1, 43, 44),
}

MRZ_NUMBERS = "0123456789"
MRZ_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MRZ_CHARACTERS = "".join([MRZ_NUMBERS, MRZ_LETTERS, "<"])

MRZ_REPLACEMENTS = {
    "0": "O",
    "O": "0",
    "D": "0",
    "I": "1",
    "1": "I",
    "2": "Z",
    "Z": "2",
    "5": "S",
    "S": "5",
    "B": "8",
    "8": "B",
}

# Helper methods for debugging


def print_if_verbose(text: str, verbose: bool):
    """Prints the text if verbose is True."""
    if verbose:
        print(text)


def display_if_verbose(image_title: str, image, verbose: bool):
    """Displays the image if verbose is True."""
    if verbose:
        display(image_title, image)
