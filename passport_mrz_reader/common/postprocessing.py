"""In this module the MRZ text is postprocessed using box heights,
MRZ fields and line lengths"""
from typing import Optional

from passport_mrz_reader.common.mrz_common import (
    MRZ_LETTERS,
    MRZ_NUMBERS,
    MRZ_REPLACEMENTS,
    PASSPORT_FIELDS,
    print_if_verbose,
)
from passport_mrz_reader.common.interfaces import (
    PostProcessors,
    PostProcessorMetadata,
)


def replace_based_on_box_heights(
    mrz_text: str, box_heights: list[float]
) -> Optional[str]:
    """Replace characters in the MRZ text based on the heights of the boxes
    that were used to get the raw MRZ text. Looks at the two MRZ lines seperately.

    Args:
        mrz_text(str): The raw MRZ text
        box_heights(list): The heights of the boxes that were used to get the
            raw MRZ text
    """
    line1, line2 = mrz_text.splitlines()[:2]
    box_heights1, box_heights2 = box_heights[:44], box_heights[44:]
    for line, box_heights in zip([line1, line2], [box_heights1, box_heights2]):
        number_heights = [
            height
            for height, letter in zip(box_heights, line)
            if letter in MRZ_NUMBERS
        ]
        letter_heights = [
            height
            for height, letter in zip(box_heights, mrz_text)
            if letter in MRZ_LETTERS
        ]
        average_number_height = (
            (sum(number_heights) / len(number_heights))
            if number_heights
            else 0
        )
        average_letter_height = (
            (sum(letter_heights) / len(letter_heights))
            if letter_heights
            else 0
        )

        if average_number_height > 0 and average_letter_height > 0:
            for index, (character, height) in enumerate(
                zip(mrz_text, box_heights), start=0 if line == 0 else 44
            ):
                if character == "\n":
                    continue
                high_letter = (
                    height > average_letter_height * 1.15
                    and character in MRZ_LETTERS
                    and character in MRZ_REPLACEMENTS
                )
                low_number = (
                    height < average_number_height * 0.85
                    and character in MRZ_NUMBERS
                    and character in MRZ_REPLACEMENTS
                )
                if high_letter or low_number:
                    mrz_text = f"{mrz_text[:index]}{MRZ_REPLACEMENTS[character]}{mrz_text[index + 1:]}"
    return mrz_text


def _replace_characters(row, start, end, mrz_lines, characters):
    """Replace characters in the MRZ text based on the different MRZ fields.
    For example, the first line should only contain letters"""
    replacements = {
        letter: number
        for letter, number in MRZ_REPLACEMENTS.items()
        if letter in characters
    }
    replaced_text = mrz_lines[row][start:end].translate(
        str.maketrans(replacements)
    )
    mrz_lines[
        row
    ] = f"{mrz_lines[row][:start]}{replaced_text}{mrz_lines[row][end:]}"
    return mrz_lines


def replace_based_on_mrz_fields(mrz_text: str) -> Optional[str]:
    """Replace characters in the MRZ text based on the different MRZ fields.
    For example, the first line should only contain letters"""
    mrz_lines = mrz_text.splitlines()
    for key, (row, start, end) in PASSPORT_FIELDS.items():
        if key in (
            "document_type",
            "country_code",
            "full name",
            "nationality",
        ):
            # replace numbers by letters
            mrz_lines = _replace_characters(
                row, start, end, mrz_lines, MRZ_NUMBERS
            )
        else:
            # Replace letters by number
            mrz_lines = _replace_characters(
                row, start, end, mrz_lines, MRZ_LETTERS
            )
    mrz_text = "\n".join(mrz_lines)
    return mrz_text


def fix_line_lengths(mrz_text: str, verbose=False) -> Optional[str]:
    """Fix the line lengths of the MRZ text ensuring it has two lines of 44
    characters."""
    # Fix line lengths
    mrz_lines = mrz_text.splitlines()
    # Remove any "ghost" lines
    mrz_lines = [line for line in mrz_lines if len(line) > 10]
    if len(mrz_lines) != 2:
        print_if_verbose("Invalid number of MRZ lines", verbose)
        return None
    # Fix first line by adding < characters at the end
    index = mrz_lines[0].find("<<<")
    if index != -1:
        trailing_characters = "".join("<" for _ in range(44 - index))
        mrz_lines[0] = f"{mrz_lines[0][:index]}{trailing_characters}"
    if len(mrz_lines[0]) != 44 or len(mrz_lines[1]) != 44:
        print_if_verbose(
            f"Incorrect number of characters ({len(mrz_lines[0])}, {len(mrz_lines[1])})",
            verbose,
        )
        return None
    return "\n".join(mrz_lines)


def postprocess(
    mrz_text: str,
    metadata: PostProcessorMetadata,
    post_processors: PostProcessors,
    verbose=False,
) -> Optional[str]:
    """Postprocess the raw MRZ text by swapping characters that are easily
    confused and making sure the lines are of the correct length.
    Should be called after get_raw_mrz_text().

    Args:
        mrz_text: The raw MRZ text
        metadata: The gathered metadata from the engine
        post_processors: The configured post-processors to use
        verbose: Whether to print debug information and display images
    """
    # Replace characters based on box heights. This must be called before
    # _fix_line_lengths() to ensure the correct box heights are used.
    if mrz_text is None:
        return None
    if post_processors.character_height is not None:
        if metadata.box_heights is None:
            print_if_verbose("No box heights available", verbose)
        else:
            mrz_text = replace_based_on_box_heights(
                mrz_text, metadata.box_heights
            )
            print_if_verbose(
                f"MRZ text after looking at box heights:\n{mrz_text}", verbose
            )

    if post_processors.line_lengths is not None:
        mrz_text = fix_line_lengths(mrz_text, verbose)
        print_if_verbose(
            f"MRZ text after looking at line lengths:\n{mrz_text}", verbose
        )
        if mrz_text is None:
            return None

    if post_processors.mrz_fields is not None:
        mrz_text = replace_based_on_mrz_fields(mrz_text)
        print_if_verbose(
            f"MRZ text after looking at MRZ fields:\n{mrz_text}", verbose
        )

    return mrz_text
