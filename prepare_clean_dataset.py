
from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "test_mock_data"
    / "aggregate_15min.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "test_mock_data"
    / "aggregate_15min_clean.csv"
)


MIN_READINGS = 50


valid_rows = []
removed_rows = 0


with open(INPUT_FILE, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        try:
            timestamp = int(row["timestamp"])
            average_power = float(row["average_power_w"])
            reading_count = int(row["reading_count"])

        except (ValueError, KeyError):
            removed_rows += 1
            continue

        # Remove windows with insufficient readings
        if reading_count < MIN_READINGS:
            removed_rows += 1
            continue

        # Remove invalid power values
        if average_power < 0:
            removed_rows += 1
            continue

        valid_rows.append({
            "timestamp": timestamp,
            "average_power_w": average_power,
            "reading_count": reading_count
        })


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "timestamp",
        "average_power_w",
        "reading_count"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(valid_rows)


print("Dataset preparation completed")
print("-" * 40)

print("Original rows:", len(valid_rows) + removed_rows)
print("Valid rows:", len(valid_rows))
print("Removed rows:", removed_rows)

print("Minimum required readings:", MIN_READINGS)

print("Output file:", OUTPUT_FILE)
print("File created:", OUTPUT_FILE.exists())