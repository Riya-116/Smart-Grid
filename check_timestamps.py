"""
check_timestamps.py
------------------------------------------------------------------
Run this to check whether smart_meter_demand.csv has real, usable
dates or got mangled by Excel (a common issue with this dataset).

    python check_timestamps.py
------------------------------------------------------------------
"""
import pandas as pd

DEMAND_CSV_PATH = r"C:\Users\HP\Downloads\smart_grid_rl_project\test_mock_data\smart_meter_demand.csv"

print("--- Raw first 5 lines (exactly as stored in the file) ---")
with open(DEMAND_CSV_PATH) as f:
    for i, line in enumerate(f):
        print(repr(line))
        if i >= 4:
            break

print("\n--- Parsed as dates ---")
df = pd.read_csv(DEMAND_CSV_PATH, nrows=1000)
df["tstp_parsed"] = pd.to_datetime(df["tstp"], errors="coerce")
print(df[["tstp", "tstp_parsed"]].head(10))

unique_dates = df["tstp_parsed"].dt.date.dropna().unique()
today = pd.Timestamp.now().date()

print(f"\nNumber of distinct dates found in first 1000 rows: {len(unique_dates)}")
if len(unique_dates) == 1 and unique_dates[0] == today:
    print("*** PROBLEM: every row parsed to TODAY'S date. ***")
    print("This means the real date/hour was lost from 'tstp' -- almost")
    print("certainly an Excel display/save artifact. Re-download a fresh")
    print("copy and do NOT open it in Excel; inspect it with a plain text")
    print("editor or this script instead.")
elif len(unique_dates) <= 2:
    print("Only 1-2 distinct dates found -- could be legitimate (a small")
    print("sample file) or could still indicate a problem. Check the raw")
    print("lines printed above: do they show a full date like")
    print("'2013-03-05 00:30:00.0000000', or just a bare time like '00:30:00'?")
else:
    print("Looks healthy -- multiple real historical dates found.")
    print(f"Date range: {min(unique_dates)} to {max(unique_dates)}")