from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "test_mock_data"
    / "house_1"
)


CHANNELS = {
    1: "aggregate",
    2: "boiler",
    5: "washing_machine",
    6: "dishwasher",
    10: "kettle",
    12: "fridge",
}


# --------------------------------------------------
# Function to inspect one channel
# --------------------------------------------------

def inspect_channel(channel_number, appliance_name):
    file_path = DATA_DIR / f"channel_{channel_number}.dat"

    print("\n" + "=" * 60)
    print(f"CHANNEL {channel_number}: {appliance_name}")
    print("=" * 60)

    if not file_path.exists():
        print("File not found:", file_path)
        return

    count = 0
    minimum_power = float("inf")
    maximum_power = float("-inf")
    total_power = 0.0

    first_timestamp = None
    last_timestamp = None

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            try:
                timestamp = int(parts[0])
                power = float(parts[1])

            except ValueError:
                continue

            if first_timestamp is None:
                first_timestamp = timestamp

            last_timestamp = timestamp

            count += 1
            total_power += power

            minimum_power = min(minimum_power, power)
            maximum_power = max(maximum_power, power)

    if count == 0:
        print("No valid measurements found.")
        return

    average_power = total_power / count

    print("File:", file_path.name)
    print("Number of readings:", count)
    print("First timestamp:", first_timestamp)
    print("Last timestamp:", last_timestamp)
    print("Minimum power:", minimum_power, "W")
    print("Maximum power:", maximum_power, "W")
    print("Average power:", round(average_power, 2), "W")


# --------------------------------------------------
# Main program
# --------------------------------------------------

print("=" * 60)
print("SMART GRID DATASET STATISTICS")
print("=" * 60)

print("Dataset directory:", DATA_DIR)
print("Dataset exists:", DATA_DIR.exists())


for channel_number, appliance_name in CHANNELS.items():
    inspect_channel(channel_number, appliance_name)