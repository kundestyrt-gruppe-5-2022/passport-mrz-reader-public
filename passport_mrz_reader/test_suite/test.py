"""Testing of image recognition"""
from typing import Optional, Any
from passport_mrz_reader.common.interfaces import Engine, PostProcessors, PreProcessors
from passport_mrz_reader.common.process import process
from passport_mrz_reader.utils.custom_passport_checker import CustomPassportChecker

SingleResult = tuple[Optional[str], bool, bool, Optional[str], str, Optional[list[str]]]
"""Tuple of (ident?, valid, correct, read_text, labeled_text, list? of reasons for invalidity)"""


def check_recognition(
    workload: tuple[
        Any, str, tuple[PreProcessors, Engine, PostProcessors], Optional[str]
    ],
    ignore_first_line: bool = True,
) -> SingleResult:
    """Performs text recognition using a workload and evaluates its correctness

    Args:
        workload: tuple of (image data, correct text, config, ident), where config is
                  (pre-processors, engine, post-processors), and ident is an optional
                  identifier which is passed directly to the result, helping the
                  caller identify which workload the result belongs to.
        ignore_first_line: ignore the validity and correctness of the first line
    Returns: the result of the recognition, a SingleResult
    """

    image, correct_text, config, ident = workload
    pre_options, engine, post_options = config

    mrz_text = process(image, pre_options, engine, post_options)

    if mrz_text is not None and len(lines := mrz_text.splitlines()) == 2:
        checker = CustomPassportChecker(
            lines[0], lines[1], ignore_first_line=ignore_first_line
        )
        valid = checker.is_correct()
        reasons = checker.get_reasons_failing()
        correct = (
            (lines[1] == correct_text.splitlines()[1])
            if ignore_first_line
            else mrz_text == correct_text
        )
    else:
        valid = False
        correct = False
        reasons = ["No result"] if mrz_text is None else ["Wrong number of lines"]

    return ident, valid, correct, mrz_text, correct_text, reasons


def accumulate(
    prev_report: tuple[list[SingleResult], int, int, int, int], new_result: SingleResult
) -> tuple[list[SingleResult], int, int, int, int]:
    """Concatenates the incoming results whilst counting (processed, valid, correct, total)
    and printing status. Use in 'reduce' of results of 'check_recognition'.

    Args:
        prev_report: tuple of (results, processed, valid, correct, total) to add result to,
        where results is a list concatenation of all new_results and the others are counters.
        new_result: SingleResult to append to results and potentially increment counters

    Returns: the updated report
    """
    _, valid, correct, _, _, _ = new_result
    (
        prev_results,
        prev_num_processed,
        prev_num_valid,
        prev_num_correct,
        num_total,
    ) = prev_report

    num_valid = prev_num_valid + int(valid)
    num_correct = prev_num_correct + int(correct)
    num_processed = prev_num_processed + 1

    print(
        "  "
        f"Processed: {num_processed}/{num_total} ({round(num_processed * 100 / num_total)}%)    "
        f"Valid: {num_valid}/{num_processed} ({round(num_valid * 100 / num_processed)}%)    "
        f"Correct: {num_correct}/{num_processed} ({round(num_correct * 100 / num_processed)}%)    ",
        end="\r",
    )

    return prev_results + [new_result], num_processed, num_valid, num_correct, num_total
