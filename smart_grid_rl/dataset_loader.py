
from pathlib import Path
from collections import defaultdict
import csv


class SmartGridDatasetLoader:

    def __init__(self, data_directory, interval_minutes=15):
        self.data_directory = Path(data_directory)
        self.interval_seconds = interval_minutes * 60

    def aggregate_channel(self, channel_number):

        file_path = self.data_directory / f"channel_{channel_number}.dat"

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        buckets = defaultdict(lambda: [0.0, 0])

        print(f"Reading: {file_path}")
        print("Please wait...")

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

                window_start = (
                    timestamp // self.interval_seconds
                ) * self.interval_seconds

                buckets[window_start][0] += power_watts
                buckets[window_start][1] += 1

        aggregated_data = []

        for timestamp in sorted(buckets):

            total_power, count = buckets[timestamp]

            if count == 0:
                continue

            average_power = total_power / count

            aggregated_data.append({
                "timestamp": timestamp,
                "average_power_w": round(average_power, 2),
                "reading_count": count
            })

        return aggregated_data

    def save_to_csv(self, aggregated_data, output_file):

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "average_power_w",
                    "reading_count"
                ]
            )

            writer.writeheader()
            writer.writerows(aggregated_data)