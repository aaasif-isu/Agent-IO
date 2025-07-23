# main.py (Definitive Final Version)
import yaml
import argparse
from pathlib import Path
import pandas as pd

from data_loader import load_all_data
from llm_api import call_llm
from agent import create_prompt
from utils import generate_suggestions_csv, extract_pipeline_data, apply_llm_suggestions_to_csv # <--- ADD apply_llm_suggestions_to_csv


def main():
    # --- Expert Knowledge Base: The Strategy & Impact Glossary ---
    feature_glossary = {
        'api': 'Controls the I/O interface. Changing from POSIX to MPIIO can improve performance for highly parallel jobs.',
        'transferSize': 'The size of each I/O operation. Larger values can increase throughput for sequential access but may use more memory.',
        'blockSize': 'The total size of a contiguous data block. Larger values are generally better for large files and sequential access.',
        'segmentCount': 'The number of data segments. Higher values can increase parallelism but also metadata overhead.',
        'numTasks': 'The number of concurrent processes. Higher values increase parallelism but can lead to contention.',
        'filePerProc': 'Using one file per process (1) can reduce contention but creates many small files. Sharing files (0) is the opposite.',
        'useStridedDatatype': 'Enables non-contiguous access. Useful for specific data patterns but can be less performant than simple sequential I/O.',
        'setAlignment': 'Aligns data in memory. Matching this to the filesystem block size is critical for performance.',
        'useO_DIRECT': 'Bypasses the OS cache (1). This can be faster for very large transfers but slower for repeated access to the same data.',
        'fsync': 'Forces writes to disk (1). This is safe but very slow. Disabling it (0) is much faster but risks data loss on a crash.',
        'LUSTRE_STRIPE_SIZE': 'The size of a data chunk on a Lustre OST. This should be tuned to match the application\'s I/O size.',
        'LUSTRE_STRIPE_WIDTH': 'The number of storage servers to stripe data across. A higher width increases parallelism but also network overhead.'
    }
    
    # --- Setup and Config ---
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data_v2'
    output_dir = base_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    with open(base_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    pipeline_choices = [p['type'] for p in config['pipelines']]
    parser = argparse.ArgumentParser(description="HPC I/O Performance Analyzer")
    parser.add_argument('pipeline', choices=pipeline_choices, help="The analysis pipeline to run.")
    args = parser.parse_args()
    print(f"Starting analysis for pipeline: {args.pipeline}")

    # --- Load Data ---
    data = load_all_data(data_dir)
    if data[-1] is None:
        print("Halting execution due to data loading failure.")
        return
    worst_job_raw, worst_job_shap, worst_job_config, worst_test_id = data



    # This dictionary holds the baseline configuration for the worst job
    original_config = worst_job_config.drop(columns=['config_id', 'testFile'], errors='ignore').iloc[0].to_dict()

    try:
        darshan_shap_suggestion_content = (output_dir / f"suggestion_darshan_shap_{worst_test_id}.txt").read_text()
        raw_darshan_suggestion_content = (output_dir / f"suggestion_raw_darshan_{worst_test_id}.txt").read_text()
        shap_only_suggestion_content = (output_dir / f"suggestion_shap_only_{worst_test_id}.txt").read_text()
    except FileNotFoundError as e:
        print(f"Error: Could not find LLM suggestion output file required for CSV generation: {e}")
        print(f"Please ensure you have run main.py for 'darshan_shap', 'raw_darshan', and 'shap_only' pipelines "
              f"for test_id '{worst_test_id}' to generate these files in the '{output_dir}' directory.")
        return # Exit if files are not found # Exit if files are not found, as CSV generation/application cannot proceed

    # Parse the suggestion content for each pipeline
    _, darshan_shap_after = extract_pipeline_data(darshan_shap_suggestion_content)
    _, raw_darshan_after = extract_pipeline_data(raw_darshan_suggestion_content)
    _, shap_only_after = extract_pipeline_data(shap_only_suggestion_content)

    # --- Automated Discovery of Optimization Levers (existing code) ---
    ior_config_df = pd.read_csv(data_dir / 'ior_configurations(in).csv')
    parameter_options = {}
    tunable_params = [col for col in ior_config_df.columns if col not in ['config_id', 'testFile']]

    for param in tunable_params:
        parameter_options[param] = sorted(ior_config_df[param].unique().tolist())

    # --- Format Config String (existing code) ---
    config_dict = worst_job_config.drop(columns=['config_id', 'testFile'], errors='ignore').iloc[0].to_dict()
    sorted_config_items = sorted(config_dict.items())
    config_string = "\n".join([f"{key} = {value}" for key, value in sorted_config_items])

    # --- Define the comprehensive deny-list (existing code) ---
    cols_to_ignore = ['nprocs', 'test_id', 'y_true', 'y_pred', 'error', 'tag']
    
    # --- Determine which data is available and build the correct prompt ---
    # And perform CSV generation conditionally based on the selected pipeline
    original_ior_config_csv_path = data_dir / 'ior_configurations(in).csv' # Path to your original config CSV

    if args.pipeline == "raw_darshan":
        job_raw_data = worst_job_raw.iloc[0].drop(labels=cols_to_ignore, errors='ignore')
        numeric_job_data = pd.to_numeric(job_raw_data, errors='coerce').dropna()
        non_zero_counters = numeric_job_data[numeric_job_data > 0]
        top_counters = non_zero_counters.nlargest(5).sort_index()
        diagnosis_summary = f"Top Raw I/O Operations:\n{top_counters.to_string()}"
        prompt = create_prompt(diagnosis_summary, config_string, has_shap=False, has_darshan=True, options=parameter_options, glossary=feature_glossary)
        
        # Only create CSVs for raw_darshan if this pipeline is selected
        generate_suggestions_csv(original_config, raw_darshan_after, "raw_darshan", output_dir, worst_test_id)
        modified_raw_darshan_csv_path = output_dir / f"ior_configurations_modified_raw_darshan_{worst_test_id}.csv"
        apply_llm_suggestions_to_csv(
            original_ior_config_csv_path,
            raw_darshan_after,
            worst_test_id,
            modified_raw_darshan_csv_path
        )

    elif args.pipeline == "darshan_shap":
        job_shap_data = worst_job_shap.drop(columns=cols_to_ignore, errors='ignore').iloc[0]
        top_shap = job_shap_data.nlargest(5).sort_index()
        job_raw_data = worst_job_raw.iloc[0].drop(labels=cols_to_ignore, errors='ignore')
        numeric_job_data = pd.to_numeric(job_raw_data, errors='coerce').dropna()
        non_zero_counters = numeric_job_data[numeric_job_data > 0]
        top_raw = non_zero_counters.nlargest(5).sort_index()
        diagnosis_summary = (f"Top Performance Bottlenecks (from SHAP analysis):\n{top_shap.to_string()}\n\n"
                           f"Top Most Frequent Raw I/O Operations (from Darshan counters):\n{top_raw.to_string()}")
        prompt = create_prompt(diagnosis_summary, config_string, has_shap=True, has_darshan=True, options=parameter_options, glossary=feature_glossary)

        # Only create CSVs for darshan_shap if this pipeline is selected
        generate_suggestions_csv(original_config, darshan_shap_after, "darshan_shap", output_dir, worst_test_id)
        modified_darshan_shap_csv_path = output_dir / f"ior_configurations_modified_darshan_shap_{worst_test_id}.csv"
        apply_llm_suggestions_to_csv(
            original_ior_config_csv_path,
            darshan_shap_after,
            worst_test_id,
            modified_darshan_shap_csv_path
        )

    elif args.pipeline == "shap_only":
        job_shap_data = worst_job_shap.drop(columns=cols_to_ignore, errors='ignore').iloc[0]
        top_shap = job_shap_data.nlargest(5).sort_index()
        diagnosis_summary = f"Top Performance Bottlenecks (from SHAP analysis):\n{top_shap.to_string()}"
        prompt = create_prompt(diagnosis_summary, config_string, has_shap=True, has_darshan=False, options=parameter_options, glossary=feature_glossary)

        # Only create CSVs for shap_only if this pipeline is selected
        generate_suggestions_csv(original_config, shap_only_after, "shap_only", output_dir, worst_test_id)
        original_ior_config_csv_path = data_dir / 'ior_configurations(in).csv'
        modified_shap_only_csv_path = output_dir / f"ior_configurations_modified_shap_only_{worst_test_id}.csv"
        apply_llm_suggestions_to_csv(
            original_ior_config_csv_path,
            shap_only_after,
            worst_test_id,
            modified_shap_only_csv_path
        )

    # --- Get LLM Suggestion and Save ---
    suggestion = call_llm(prompt)
    output_file = output_dir / f"suggestion_{args.pipeline}_{worst_test_id}.txt"
    with open(output_file, 'w') as f:
        f.write(f"--- Analysis for test_id: {worst_test_id} ---\n")
        f.write(f"--- Pipeline: {args.pipeline} ---\n")
        f.write("\n--- PROMPT SENT TO LLM ---\n")
        f.write(prompt)
        f.write("\n\n" + "="*50 + "\n\n")
        f.write("--- LLM SUGGESTION ---\n")
        f.write(suggestion)
    print(f"\nSuccess! Suggestion saved to: {output_file}")

if __name__ == '__main__':
    main()