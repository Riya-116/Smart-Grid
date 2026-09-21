from pathlib import Path
from collections import Counter


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "test_mock_data"
    / "house_1"
)


CHANNELS = [1, 2, 5, 6, 10, 12]

MAX_INTERVALS = 100_000


def analyze_channel(channel_number):
    file_path = DATA_DIR / f"channel_{channel_number}.dat"

    print("\n" + "=" * 60)
    print(f"CHANNEL {channel_number}")
    print("=" * 60)

    previous_timestamp = None
    interval_counts = Counter()

    valid_readings = 0
    invalid_readings = 0

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            if valid_readings >= MAX_INTERVALS:
                break

            parts = line.strip().split()

            if len(parts) != 2:
                invalid_readings += 1
                continue

            try:
                timestamp = int(parts[0])
                float(parts[1])

            except ValueError:
                invalid_readings += 1
                continue

            if previous_timestamp is not None:

                interval = timestamp - previous_timestamp

                if interval >= 0:
                    interval_counts[interval] += 1

            previous_timestamp = timestamp
            valid_readings += 1

    print("Readings inspected:", valid_readings)
    print("Invalid readings:", invalid_readings)

    print("\nMost common timestamp intervals:")

    for interval, count in interval_counts.most_common(10):
        print(f"{interval} seconds: {count} occurrences")


for channel in CHANNELS:
    analyze_channel(channel)
    