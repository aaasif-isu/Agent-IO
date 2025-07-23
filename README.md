# HPC I/O Performance Analyzer

This project analyzes HPC I/O data, generates LLM-based optimization suggestions, and creates structured CSV reports.

---

## Setup

1.  **Clone the repository/Organize files**: Ensure your project structure matches:
    ```
    agent_io/
    ├── code_v2/
    │   ├── main.py
    │   ├── agent.py
    │   ├── data_loader.py
    │   ├── llm_api.py
    │   ├── utils.py # New file
    │   ├── config.yaml
    │   └── data_v2/
    │   └── output/
    └── ...
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas pyyaml
    ```

---

## How to Run

Navigate to the `code_v2` directory in your terminal.

### Step 1: Generate LLM Suggestion Text Files (.txt)

Run the script for each analysis pipeline. This calls the LLM and saves its full suggestions as `.txt` files in `output/`. You **must** run each of these at least once:

```bash
python main.py darshan_shap
python main.py raw_darshan
python main.py shap_only