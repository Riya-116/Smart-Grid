"""
run_real_data.py
------------------------------------------------------------------
Train and evaluate the Smart Grid Reinforcement Learning project
using the available Kaggle solar and UK-DALE datasets.

Run:
    py run_real_data.py
------------------------------------------------------------------
"""

from train import train
from experiments import compare


# ---------------------------------------------------------------
# REAL DATA CONFIGURATION
# ---------------------------------------------------------------

REAL_DATA = dict(

    # -----------------------------------------------------------
    # KAGGLE SOLAR DATASET
    # -----------------------------------------------------------

    kaggle_solar_csv=(
        r"C:\Users\HP\Downloads\smart_grid_rl_project"
        r"\test_mock_data\plant1_generation.csv"
    ),

    # -----------------------------------------------------------
    # UK-DALE DATASET
    # -----------------------------------------------------------

    ukdale_house_dir=(
        r"C:\Users\HP\Downloads\smart_grid_rl_project"
        r"\test_mock_data\house_1"
    ),

    # Appliance names to load from the UK-DALE dataset.
    #
    # The environment expects ukdale_appliance_names,
    # not ukdale_appliance_map.
    ukdale_appliance_names=[
        "washing_machine",
    ],

    # -----------------------------------------------------------
    # HOUSEHOLD SOLAR CONFIGURATION
    # -----------------------------------------------------------

    household_solar_capacity_kw=4.0,

    # -----------------------------------------------------------
    # SMART METER AND TARIFF DATA
    # -----------------------------------------------------------

    # No smart_meter_demand_csv is included because the
    # environment does not accept that parameter.

    # No tariff CSV is available.
    # The environment will use synthetic electricity prices.
)


# ---------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------
    # TRAIN THE Q-LEARNING AGENT
    # -----------------------------------------------------------

    print("=== Training on REAL data ===")

    train(
        n_episodes=3000,
        save_path="q_table_real.pkl",
        seed=1,
        env_kwargs=REAL_DATA,
    )

    # -----------------------------------------------------------
    # COMPARE RL CONTROLLER WITH BASELINE CONTROLLER
    # -----------------------------------------------------------

    print("\n=== Comparing RL vs baseline on REAL data ===")

    compare(
        n_test_days=20,
        q_table_path="q_table_real.pkl",
        seed=42,
        env_kwargs=REAL_DATA,
    )