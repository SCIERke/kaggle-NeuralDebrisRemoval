"""
Shared submission-writing logic for kaggle/submit.py and kaggle/full_pipeline.py.
"""

import csv

from config.config import SAMPLE_SUBMISSION_PATH


def write_submission(
    predictions: dict[int, str], out_path: str, sample_submission_path: str = SAMPLE_SUBMISSION_PATH
) -> None:
    """Mirrors sample_submission.csv's exact row order and id/image_id
    pairing instead of inventing our own (e.g. sorting image_id numerically).
    sample_submission.csv's image_id column is in lexicographic (string)
    file-listing order, not numeric order — assigning "id" by numeric sort
    diverges from that pairing starting at id=2, which produced a real
    "image_id values not present in the solution" rejection even though
    every image_id was present, just paired with the wrong id."""
    with open(sample_submission_path) as f_in, open(out_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["id", "image_id", "prediction_string"])
        n = 0
        for row in reader:
            image_id = int(row["image_id"])
            # competition requires a literal " " for no-detection rows —
            # an empty string is treated as null by Kaggle's csv parser.
            writer.writerow([row["id"], image_id, predictions.get(image_id, "") or " "])
            n += 1
    print(f"Submission written → {out_path}  ({n} rows, mirroring {sample_submission_path}'s row order)")
