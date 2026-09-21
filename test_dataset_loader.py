
from pathlib import Path

from smart_grid_rl.dataset_loader import SmartGridDatasetLoader


PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    PROJECT_DIR
    / "test_mock_data"
    / "house_1"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "test_mock_data"
    / "aggregate_15min.csv"
)


loader = SmartGridDatasetLoader(
    DATA_DIR,
    interval_minutes=15
)


aggregated_data = loader.aggregate_channel(1)


print(
    "Number of aggregated windows:",
    len(aggregated_data)
)


print("\nFirst 5 records:")

for record in aggregated_data[:5]:
    print(record)


loader.save_to_csv(
    aggregated_data,
    OUTPUT_FILE
)


print("\nSaved file:", OUTPUT_FILE)
print("File exists:", OUTPUT_FILE.exists())