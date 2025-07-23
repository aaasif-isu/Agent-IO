# utils.py
import pandas as pd
from pathlib import Path
import re
import numpy as np # <--- ADD THIS IMPORT for numeric type checking

# (No PARAMETER_TYPES dictionary needed anymore here)

def parse_config_block(block_str: str) -> dict:
    """
    Parses a configuration block string (e.g., from LLM output) into a dictionary.
    """
    config = {}
    for line in block_str.strip().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
    return config

def parse_after_block(block_str: str) -> dict:
    """
    Parses the 'After' configuration block from LLM output, extracting parameter,
    suggested value, impact, and risk scores.
    """
    suggested_changes = {}
    lines = block_str.strip().split('\n')
    for line in lines:
        # Regex to capture parameter, value, impact, risk, and optional justification
        match = re.match(r'(\w+) = (.+?)\s+\(Impact: (\d+), Risk: (\d+)\s+-\s*(.+?)\)', line)
        if match:
            param, value, impact, risk, _ = match.groups()
            suggested_changes[param.strip()] = {
                'value': value.strip(),
                'impact': int(impact),
                'risk': int(risk)
            }
        else:
            # Handle cases where only a simple parameter = value is present
            match_simple = re.match(r'(\w+) = (.+)', line)
            if match_simple:
                param, value = match_simple.groups()
                suggested_changes[param.strip()] = {
                    'value': value.strip(),
                    'impact': 'N/A', # No impact/risk provided
                    'risk': 'N/A'    # No impact/risk provided
                }
    return suggested_changes

def extract_pipeline_data(content: str) -> tuple[dict, dict]:
    """
    Extracts the 'Before' and 'After' configuration blocks from an LLM pipeline
    suggestion content string. Returns (before_config_dict, after_changes_dict).
    """
    before_match = re.search(r'\*\*Before:\*\*(.*?)\*\*After:\*\*', content, re.DOTALL)
    after_match = re.search(r'\*\*After:\*\*(.*)', content, re.DOTALL)

    before_config = {}
    if before_match:
        before_config = parse_config_block(before_match.group(1))

    after_changes = {}
    if after_match:
        after_changes = parse_after_block(after_match.group(1))

    return before_config, after_changes

def generate_suggestions_csv(original_config: dict,
                             llm_suggestions: dict,
                             pipeline_name: str,
                             output_dir: Path,
                             test_id: str):
    """
    Generates a CSV file comparing original config with LLM-suggested changes for a specific pipeline.

    Args:
        original_config (dict): The original I/O configuration (parameter: value).
        llm_suggestions (dict): Dictionary of suggested changes from a specific
                                 LLM pipeline (parameter: {'value': ..., 'impact': ..., 'risk': ...}).
        pipeline_name (str): The name of the pipeline (e.g., "darshan_shap"),
                             used for column naming and filename.
        output_dir (Path): The directory where the CSV file will be saved.
        test_id (str): The ID of the job being analyzed, used for the filename.
    """
    data_for_df = []

    for param, original_value in original_config.items():
        row = {
            'Parameter': param,
            'Original Value': original_value,
            f'{pipeline_name} Suggested Value': original_value, # Default to original value
            f'{pipeline_name} Impact': 'N/A', # Default to N/A
            f'{pipeline_name} Risk': 'N/A'    # Default to N/A
        }
        data_for_df.append(row)

    df = pd.DataFrame(data_for_df)
    df.set_index('Parameter', inplace=True) # Set 'Parameter' as index for easy updating

    for param, details in llm_suggestions.items():
        # Skip generic placeholder parameters like 'parameter1', 'parameter2'
        if param.startswith('parameter'):
            continue

        if param in df.index:
            df.loc[param, f'{pipeline_name} Suggested Value'] = details['value']
            df.loc[param, f'{pipeline_name} Impact'] = details['impact']
            df.loc[param, f'{pipeline_name} Risk'] = details['risk']
        else:
            # Handle cases where the LLM suggests a parameter not present in the original configuration
            new_row = {
                'Original Value': 'N/A', # Indicates this parameter wasn't in the original config
                f'{pipeline_name} Suggested Value': details['value'],
                f'{pipeline_name} Impact': details['impact'],
                f'{pipeline_name} Risk': details['risk']
            }
            df.loc[param] = new_row # Add the new row to the DataFrame

    df.reset_index(inplace=True) # Convert 'Parameter' index back to a regular column for CSV export

    # Define output file path and save the DataFrame to CSV
    output_file = output_dir / f"comparison_suggestions_{pipeline_name}_{test_id}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nCSV comparison for '{pipeline_name}' suggestions saved to: {output_file}")

PARAMETER_TYPES = {
    'api': str,
    'transferSize': str,
    'blockSize': str,
    'segmentCount': int,
    'numTasks': int,
    'filePerProc': int,
    'useStridedDatatype': int,
    'setAlignment': str,
    'useO_DIRECT': int,
    'fsync': int,
    'LUSTRE_STRIPE_SIZE': str,
    'LUSTRE_STRIPE_WIDTH': int
}

def apply_llm_suggestions_to_csv(original_csv_path: Path,
                                 llm_suggested_changes: dict,
                                 target_test_id: str,
                                 output_modified_csv_path: Path):
    """
    Reads an original IOR configuration CSV, applies LLM suggested changes
    to a specific test_id's row, and saves ONLY THE MODIFIED ROW(S) to a new CSV.

    Args:
        original_csv_path (Path): Path to the original ior_configurations(in).csv.
        llm_suggested_changes (dict): Dictionary of suggested changes from LLM
                                      (e.g., {'transferSize': {'value': '1M', ...}}).
        target_test_id (str): The 'testFile' ID of the row to modify.
        output_modified_csv_path (Path): Path to save the new CSV file with applied changes.
    """
    df = pd.read_csv(original_csv_path)

    # Find the row(s) corresponding to the target_test_id
    row_indices = df[df['testFile'] == target_test_id].index

    if not row_indices.empty:
        # Apply changes to the entire DataFrame first
        for param, details in llm_suggested_changes.items():
            # Skip generic placeholder parameters
            if param.startswith('parameter'):
                continue

            if param in df.columns:
                target_dtype = df[param].dtype
                try:
                    if pd.api.types.is_integer_dtype(target_dtype) or pd.api.types.is_bool_dtype(target_dtype):
                        df.loc[row_indices, param] = int(details['value'])
                    else:
                        df.loc[row_indices, param] = str(details['value'])
                except ValueError:
                    print(f"Warning: Could not convert suggested value '{details['value']}' "
                          f"to {target_dtype} for parameter '{param}' of {target_test_id}. Assigning as is.")
                    df.loc[row_indices, param] = details['value']
            else:
                print(f"Warning: Suggested parameter '{param}' not found in original CSV columns. Skipping for {target_test_id}.")
        
        # --- NEW: Filter the DataFrame to contain ONLY the modified row(s) ---
        df_modified_row_only = df[df['testFile'] == target_test_id]

        df_modified_row_only.to_csv(output_modified_csv_path, index=False)
        print(f"Modified configuration for {target_test_id} saved to: {output_modified_csv_path}")
    else:
        print(f"Error: Test ID '{target_test_id}' not found in '{original_csv_path}'. No changes applied.")






def apply_llm_suggestions_to_csv_old(original_csv_path: Path,
                                 llm_suggested_changes: dict,
                                 target_test_id: str,
                                 output_modified_csv_path: Path):
    """
    Reads an original IOR configuration CSV, applies LLM suggested changes
    to a specific test_id's row, and saves the modified data to a new CSV.

    Args:
        original_csv_path (Path): Path to the original ior_configurations(in).csv.
        llm_suggested_changes (dict): Dictionary of suggested changes from LLM
                                      (e.g., {'transferSize': {'value': '1M', ...}}).
        target_test_id (str): The 'testFile' ID of the row to modify.
        output_modified_csv_path (Path): Path to save the new CSV file with applied changes.
    """
    df = pd.read_csv(original_csv_path)

    # Find the row(s) corresponding to the target_test_id
    row_indices = df[df['testFile'] == target_test_id].index

    if not row_indices.empty:
        # Apply each suggested change to the identified row(s)
        for param, details in llm_suggested_changes.items():
            # Skip generic placeholder parameters
            if param.startswith('parameter'):
                continue

            # Check if the parameter exists as a column in the DataFrame
            if param in df.columns:
                # Dynamically infer the target column's data type
                target_dtype = df[param].dtype
                
                # Explicitly cast the value based on the inferred dtype
                try:
                    if pd.api.types.is_integer_dtype(target_dtype) or pd.api.types.is_bool_dtype(target_dtype):
                        # Attempt to convert to int if it's an integer or boolean column
                        # (boolean columns often store 0/1 which can be converted to int)
                        df.loc[row_indices, param] = int(details['value'])
                    else: # Assume string/object type for others
                        df.loc[row_indices, param] = str(details['value'])
                except ValueError:
                    print(f"Warning: Could not convert suggested value '{details['value']}' "
                          f"to {target_dtype} for parameter '{param}' of {target_test_id}. Assigning as is.")
                    df.loc[row_indices, param] = details['value'] # Assign as is if conversion fails
            else:
                print(f"Warning: Suggested parameter '{param}' not found in original CSV columns. Skipping for {target_test_id}.")
        
        df.to_csv(output_modified_csv_path, index=False)
        print(f"Modified configuration for {target_test_id} saved to: {output_modified_csv_path}")
    else:
        print(f"Error: Test ID '{target_test_id}' not found in '{original_csv_path}'. No changes applied.")