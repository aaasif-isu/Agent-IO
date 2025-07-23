import pandas as pd
from pathlib import Path

def load_all_data(data_dir: Path):
    """
    Loads all necessary data files, finds the worst-performing job,
    and returns all relevant data for that job.

    Args:
        data_dir (Path): The path to the 'data' directory.

    Returns:
        A tuple containing the raw data, SHAP data, IOR config data, and the worst test ID.
        Returns (None, None, None, None) on error.
    """
    try:
        # Define the exact file names
        sorted_jobs_file = data_dir / 'darshan_parsed_output_6-29-V5_sorted_by_tag(in).csv'
        raw_darshan_file = data_dir / 'darshan_parsed_output_6-29-V5(in).csv'
        shap_values_file = data_dir / 'darshan_parsed_output_6-29-V5_norm_log_scaled_with_shap_calib(in).csv'
        ior_config_file = data_dir / 'ior_configurations(in).csv'

        # Load all datasets
        sorted_df = pd.read_csv(sorted_jobs_file)
        raw_darshan_df = pd.read_csv(raw_darshan_file)
        shap_df = pd.read_csv(shap_values_file)
        ior_config_df = pd.read_csv(ior_config_file)

    except FileNotFoundError as e:
        print(f"Error: {e}.")
        print("Please ensure all four CSV files are in the 'data' directory.")
        return None, None, None, None

    # 1. Get the test_id of the worst job from the sorted file
    worst_test_id = sorted_df.iloc[2]['test_id']
    print(f"Data loaded successfully.")
    print(f"Identified worst-performing job. test_id: {worst_test_id}")

    # 2. Find the data for this specific job in each dataframe
    worst_job_raw = raw_darshan_df[raw_darshan_df['test_id'] == worst_test_id]
    worst_job_shap = shap_df[shap_df['test_id'] == worst_test_id]
    
    # --- THIS IS THE CORRECTED LINE ---
    # The column name in ior_configurations(in).csv is 'testFile'
    worst_job_config = ior_config_df[ior_config_df['testFile'] == worst_test_id]
    
    # --- Data Validation ---
    if worst_job_raw.empty or worst_job_shap.empty or worst_job_config.empty:
        print(f"Error: Could not find data for test_id '{worst_test_id}' in all files.")
        return None, None, None, None

    return worst_job_raw, worst_job_shap, worst_job_config, worst_test_id