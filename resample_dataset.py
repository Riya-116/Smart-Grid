from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "test_mock_data"
    / "house_1"
)

INPUT_FILE = DATA_DIR / "channel_1.dat"

INTERVAL_SECONDS = 15 * 60  # 15 minutes


# --------------------------------------------------
# Aggregate readings into time windows
# --------------------------------------------------

def aggregate_channel(file_path):
    buckets = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split()

            if len(parts) != 2:
                continue

            try:
                timestamp = int(parts[0])
                power_watts = float(parts[1])

            except ValueError:
                continue

            # Align timestamp to a 15-minute window
            window_start = (
                timestamp // INTERVAL_SECONDS
            ) * INTERVAL_SECONDS

            buckets[window_start].append(power_watts)

    return buckets


# --------------------------------------------------
# Main program
# --------------------------------------------------

print("=" * 60)
print("15-MINUTE AGGREGATION TEST")
print("=" * 60)

print("Input file:", INPUT_FILE)
print("File exists:", INPUT_FILE.exists())

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)


buckets = aggregate_channel(INPUT_FILE)

print("Number of 15-minute windows:", len(buckets))


print("\nFirst 10 aggregated windows:")
print("-" * 60)

for index, (timestamp, values) in enumerate(
    sorted(buckets.items())
):

    if index >= 10:
        break

    average_power = sum(values) / len(values)

    readable_time = datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc,
    )

    print(
        f"Window: {readable_time}"
    )

    print(
        f"Readings: {len(values)}"
    )

    print(
        f"Average power: {average_power:.2f} W"
    )

    print("-" * 60)
    