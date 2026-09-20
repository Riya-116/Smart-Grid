"""
data_integration.py
------------------------------------------------------------------
NEW FILE -- connects the environment to the four real datasets named
in the abstract:

    UK-DALE                       -> appliance status + demand
    Smart Meters in London        -> dynamic electricity price signal
    Kaggle Solar Power Generation -> solar generation (short-term, real)
    NREL Solar Data (NSRDB)       -> solar generation (long-term training)

Each parser is written against the dataset's actual published file
format. VirtualSmartHomeEnv runs at an arbitrary `steps_per_day`
(default 96, i.e. 15-min steps); these real sources are naturally
hourly, so `resample_to_steps()` linearly interpolates an hourly array
up (or down) to whatever resolution the environment needs.

If a file is missing or malformed, each loader raises a clear
exception; DatasetBundle.load() catches that, prints why, and leaves
the corresponding field as None so VirtualSmartHomeEnv falls back to
its built-in synthetic generator for that piece only.
------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Low-level parsers, one per dataset's real file format
# ------------------------------------------------------------------

def _read_ukdale_channel(house_dir: str, channel_number: int) -> pd.Series:
    path = f"{house_dir}/channel_{channel_number}.dat"
    df = pd.read_csv(path, sep=" ", header=None, names=["unix_time", "power_w"])
    df["timestamp"] = pd.to_datetime(df["unix_time"], unit="s")
    return df.set_index("timestamp")["power_w"]


def load_ukdale_aggregate(house_dir: str, mains_channel: int = 1) -> pd.Series:
    """channel_1 is conventionally the whole-house 'mains' aggregate demand."""
    power_w = _read_ukdale_channel(house_dir, mains_channel)
    return power_w.resample("1h").mean().interpolate() / 1000.0  # -> hourly kW


def load_ukdale_appliance(house_dir: str, appliance_name: str, on_threshold_w: float = 10.0):
    """Returns (rated_power_kw, typical_duration_h) for a named appliance,
    looked up via labels.dat."""
    labels_path = f"{house_dir}/labels.dat"
    labels = {}
    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                labels[parts[1].lower()] = int(parts[0])

    if appliance_name.lower() not in labels:
        raise ValueError(f"Appliance '{appliance_name}' not found in {labels_path}. "
                          f"Available: {list(labels.keys())}")

    channel = labels[appliance_name.lower()]
    power_w = _read_ukdale_channel(house_dir, channel)
    on_mask = power_w > on_threshold_w
    rated_power_kw = power_w[on_mask].mean() / 1000.0 if on_mask.any() else 0.0

    run_lengths, run_len = [], 0
    for is_on in on_mask:
        if is_on:
            run_len += 1
        elif run_len > 0:
            run_lengths.append(run_len)
            run_len = 0
    step_hours = power_w.index.to_series().diff().median().seconds / 3600
    typical_duration_h = float(np.median(run_lengths) * step_hours) if run_lengths else 1.0

    return round(float(rated_power_kw), 3), round(typical_duration_h, 2)


_TOU_BAND_PRICE_GBP = {"Low": 0.0399, "Normal": 0.1176, "High": 0.6720}


def load_smart_meters_price(tariff_csv_path: str) -> pd.Series:
    """TariffDateTime, Tariff (High/Normal/Low) -> hourly $/kWh series."""
    df = pd.read_csv(tariff_csv_path)
    dt_col = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
    band_col = next(c for c in df.columns if "tariff" in c.lower() and c != dt_col)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df["price"] = df[band_col].map(_TOU_BAND_PRICE_GBP)
    if df["price"].isna().any():
        raise ValueError("Unrecognized tariff band label; expected High/Normal/Low")
    series = df.set_index(dt_col)["price"]
    return series.resample("1h").mean().interpolate()


def load_smart_meters_demand(csv_path: str, lclid: Optional[str] = None) -> pd.Series:
    """LCLid, tstp, energy(kWh/hh) half-hourly -> hourly kW series."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    energy_col = next(c for c in df.columns if "energy" in c.lower() or "kwh" in c.lower())
    df["tstp"] = pd.to_datetime(df["tstp"])
    if lclid is not None and "LCLid" in df.columns:
        df = df[df["LCLid"] == lclid]
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
    series = df.set_index("tstp")[energy_col]
    return series.resample("1h").sum().interpolate()  # half-hourly kWh -> hourly kW


def load_kaggle_solar(generation_csv_path: str, household_capacity_kw: float = 4.0) -> pd.Series:
    """DATE_TIME, PLANT_ID, SOURCE_KEY, AC_POWER (multi-inverter, 15-min)
    -> hourly kW scaled down to a household rooftop system size."""
    df = pd.read_csv(generation_csv_path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], dayfirst=True, errors="coerce")
    plant_output = df.groupby("DATE_TIME")["AC_POWER"].sum()
    plant_output = plant_output.sort_index().resample("1h").mean().interpolate()
    peak = plant_output.max()
    if peak <= 0:
        raise ValueError("Kaggle solar file has no positive AC_POWER readings")
    return plant_output / peak * household_capacity_kw


def load_nrel_solar(csv_path: str, system_capacity_kw: float = 4.0, panel_derate: float = 0.8) -> pd.Series:
    """NSRDB export (2 metadata header rows) -> hourly kW via a simple
    irradiance-scaled PV model."""
    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = [c.strip() for c in df.columns]
    if "GHI" not in df.columns:
        raise ValueError("Expected an NREL NSRDB export with a 'GHI' column")
    df["timestamp"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"],
             hour=df["Hour"], minute=df.get("Minute", 0))
    )
    ghi = pd.to_numeric(df["GHI"], errors="coerce").fillna(0.0)
    solar_kw = (ghi / 1000.0) * system_capacity_kw * panel_derate
    series = pd.Series(solar_kw.values, index=df["timestamp"]).sort_index()
    return series.resample("1h").mean().interpolate()


# ------------------------------------------------------------------
# Resampling: hourly real data -> arbitrary environment step resolution
# ------------------------------------------------------------------

def resample_to_steps(hourly_values: np.ndarray, steps_per_day: int) -> np.ndarray:
    """Linearly interpolates a 24-length hourly array (or any 24*N-hour
    array) up/down to `steps_per_day` evenly spaced samples per day."""
    hourly_values = np.asarray(hourly_values, dtype=float)
    n_hours = len(hourly_values)
    hour_positions = np.linspace(0, 24, n_hours, endpoint=False)
    step_positions = np.linspace(0, 24, steps_per_day, endpoint=False)
    # wrap-around interpolation so the day is treated as cyclic
    extended_x = np.concatenate([hour_positions, [24.0]])
    extended_y = np.concatenate([hourly_values, [hourly_values[0]]])
    return np.interp(step_positions, extended_x, extended_y)


# ------------------------------------------------------------------
# Bundle: tries each real source, falls back to None on failure
# ------------------------------------------------------------------

@dataclass
class DatasetBundle:
    """Loaded (or attempted) real-data arrays, resampled to steps_per_day.
    Any field left None means: use the environment's synthetic default
    for that piece."""
    steps_per_day: int
    renewable_profile_kw: Optional[np.ndarray] = None
    price_profile: Optional[np.ndarray] = None
    appliance_overrides: dict = field(default_factory=dict)  # name -> (rated_kw, duration_h)

    @staticmethod
    def _try(loader_fn, label, path):
        if not path:
            return None
        try:
            result = loader_fn()
            print(f"[data_integration] Using REAL {label} from {path}")
            return result
        except Exception as e:
            print(f"[data_integration] Could not load {label} from {path} ({e}); "
                  f"falling back to synthetic.")
            return None

    @classmethod
    def load(
        cls,
        steps_per_day: int,
        ukdale_house_dir: Optional[str] = None,
        ukdale_mains_channel: int = 1,
        ukdale_appliance_names: Optional[list] = None,
        smart_meter_tariff_csv: Optional[str] = None,
        kaggle_solar_csv: Optional[str] = None,
        nrel_solar_csv: Optional[str] = None,
        use_nrel_for_solar: bool = False,
        household_solar_capacity_kw: float = 6.0,
    ) -> "DatasetBundle":
        bundle = cls(steps_per_day=steps_per_day)

        # Solar/renewable: Kaggle (short-term) by default, or NREL (long-term)
        if use_nrel_for_solar:
            hourly_solar = cls._try(
                lambda: load_nrel_solar(nrel_solar_csv, household_solar_capacity_kw).values,
                "NREL solar (long-term)", nrel_solar_csv)
        else:
            hourly_solar = cls._try(
                lambda: load_kaggle_solar(kaggle_solar_csv, household_solar_capacity_kw).values,
                "Kaggle solar (short-term)", kaggle_solar_csv)
        if hourly_solar is not None and len(hourly_solar) >= 24:
            bundle.renewable_profile_kw = resample_to_steps(hourly_solar[:24], steps_per_day)

        # Price: Smart Meters London ToU tariff
        hourly_price = cls._try(
            lambda: load_smart_meters_price(smart_meter_tariff_csv).values,
            "Smart Meters London price", smart_meter_tariff_csv)
        if hourly_price is not None and len(hourly_price) >= 24:
            bundle.price_profile = resample_to_steps(hourly_price[:24], steps_per_day)

        # Appliance ratings/durations: UK-DALE
        if ukdale_house_dir and ukdale_appliance_names:
            for name in ukdale_appliance_names:
                result = cls._try(
                    lambda n=name: load_ukdale_appliance(ukdale_house_dir, n),
                    f"UK-DALE appliance profile ({name})", ukdale_house_dir)
                if result is not None:
                    bundle.appliance_overrides[name] = result

        return bundle
