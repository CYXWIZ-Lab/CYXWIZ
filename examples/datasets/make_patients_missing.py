"""Build patients_missing.csv: a small made-up table with gaps, for the
missing-values example (examples/cyxgraph/missing_values_fix.cyxgraph).

Training refuses empty cells and NaN; the example shows the one-click fix in
the "Cannot start training" report. The values are generated (no real
patients): a risk class 0, 1 or 2 from age, body mass and blood pressure.

    python examples/datasets/make_patients_missing.py [output path]

Standard library only; the same seed gives the same file.
"""

import csv
import os
import random
import sys

ROWS = 300


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "patients_missing.csv")
    rng = random.Random(20261121)
    rows = []
    for _ in range(ROWS):
        age = rng.randint(22, 85)
        bmi = round(rng.gauss(27.0, 4.5), 1)
        pressure = round(rng.gauss(118 + 0.4 * (age - 40), 12.0), 1)
        cholesterol = round(rng.gauss(190 + 0.6 * (age - 40), 25.0), 1)
        smoker = 1 if rng.random() < 0.28 else 0
        score = 0.03 * (age - 50) + 0.12 * (bmi - 27) + 0.04 * (pressure - 125) + 0.8 * smoker + rng.gauss(0, 0.6)
        diagnosis = 0 if score < -0.4 else (1 if score < 0.9 else 2)
        rows.append([age, bmi, pressure, cholesterol, smoker, diagnosis])

    # The gaps: empty cells in two feature columns, "NaN" in another, and a
    # few rows without a label.
    for r in rng.sample(range(ROWS), 18):
        rows[r][0] = ""
    for r in rng.sample(range(ROWS), 9):
        rows[r][3] = ""
    for r in rng.sample(range(ROWS), 4):
        rows[r][1] = "NaN"
    for r in rng.sample(range(ROWS), 5):
        rows[r][5] = ""

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["age", "bmi", "blood_pressure", "cholesterol", "smoker", "diagnosis"])
        writer.writerows(rows)
    print(f"wrote {out}: {ROWS} rows")


if __name__ == "__main__":
    main()
