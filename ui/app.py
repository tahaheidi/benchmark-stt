import streamlit as st
import yaml
import subprocess
import os
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Configuration ---
DOCKER_COMPOSE_FILE = Path(__file__).parent.parent / "docker-compose.yml"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# --- Helper Functions ---

def get_available_datasets(compose_file):
    """Parses the docker-compose.yml to find services with profiles (our datasets)."""
    try:
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        datasets = []
        for service, config in compose_config.get('services', {}).items():
            if 'profiles' in config and service.startswith('worker_'):
                datasets.extend(config['profiles'])
        return sorted([d for d in datasets if not d.startswith('_')])
    except FileNotFoundError:
        st.error(f"Error: docker-compose.yml not found at {compose_file}")
        return []
    except Exception as e:
        st.error(f"Error parsing docker-compose.yml: {e}")
        return []

def launch_benchmark(datasets, max_samples):
    """Constructs and runs the docker compose command, showing output."""
    if not datasets:
        st.warning("Please select at least one dataset to benchmark.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    st.info("Stopping previous benchmark containers (if any)...")
    try:
        # Use a placeholder for live output
        with st.expander("Docker Compose Down Logs"):
            with st.spinner("Running `docker compose down`..."):
                result = subprocess.run(
                    ["docker", "compose", "down"],
                    cwd=DOCKER_COMPOSE_FILE.parent,
                    capture_output=True,
                    text=True,
                    check=False # Don't raise exception on non-zero exit
                )
                st.code(result.stdout + result.stderr, language='log')
        if result.returncode == 0:
            st.success("`docker compose down` completed successfully.")
        else:
            st.warning("`docker compose down` finished with an error (this is often safe to ignore).")

    except Exception as e:
        st.error(f"Failed to run 'docker compose down': {e}")
        return

    st.info("Launching new benchmark...")
    command = ["docker", "compose"]
    for dataset in datasets:
        command.extend(["--profile", dataset])
    # Run in detached mode to avoid stream blocking and network teardown issues
    command.extend(["up", "--build", "-d", "--remove-orphans"])

    st.session_state.benchmark_running = True
    st.session_state.last_benchmark_start_time = datetime.now()

    # Prepare environment for subprocess
    env = os.environ.copy()
    if max_samples:
        env["MAX_SAMPLES"] = str(max_samples)

    with st.expander("Benchmark Launch Logs", expanded=True):
        try:
            process = subprocess.Popen(command, cwd=DOCKER_COMPOSE_FILE.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            st.session_state.benchmark_process = process
            
            log_placeholder = st.empty()
            logs = ""
            for line in iter(process.stdout.readline, ''):
                logs += line
                log_placeholder.code(logs, language='log')
            
            process.wait()
            if process.returncode == 0:
                st.success(f"Benchmark for {', '.join(datasets)} completed.")
            else:
                st.error(f"Benchmark for {', '.join(datasets)} failed.")

        except FileNotFoundError:
            st.error("Error: 'docker' command not found. Is Docker installed and in your PATH?")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    st.session_state.benchmark_running = False

def load_results():
    """Load all .csv files from the results directory into a single DataFrame."""
    all_files = list(RESULTS_DIR.glob('*.csv'))
    if not all_files:
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            pass # Ignore empty files, as they might be partially written
        except Exception as e:
            print(f"Error reading {f}: {e}") # Log to console
    
    if not df_list:
        return pd.DataFrame()
    
    return pd.concat(df_list, ignore_index=True)


# --- UI Layout ---
st.set_page_config(layout="wide")

st.title("Speech-to-Text Benchmark Dashboard")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Configuration")
    st.markdown("**Model**")
    st.markdown("> Gemini 2.5 flash lite")
    
    st.markdown("**Datasets**")
    available_datasets = get_available_datasets(DOCKER_COMPOSE_FILE)
    selected_datasets = st.multiselect(
        label="Select datasets to run",
        options=available_datasets,
        label_visibility="collapsed"
    )

    max_samples = st.number_input("Max samples per dataset", min_value=1, value=15, step=1)

    if st.button("Benchmark Now", use_container_width=True):
        launch_benchmark(selected_datasets, max_samples)


# --- Main Panel for Results ---
if 'last_benchmark_start_time' in st.session_state:
    st.info(f"Last benchmark started: {st.session_state.last_benchmark_start_time.strftime('%d %B %Y %I:%M:%S %p')}")
else:
    st.info("No benchmark has been run in this session.")

st.divider()
st.subheader("Results Overview")
results_placeholder = st.empty()

# --- Live Update Loop ---
while True:
    df = load_results()
    
    if not df.empty:
        # Calculate metrics
        # Ensure required columns exist
        required_cols = ['dataset', 'wer', 'rfx']
        if not all(col in df.columns for col in required_cols):
            results_placeholder.warning("Results file(s) are missing required columns (dataset, wer, rfx). Waiting for more data...")
        else:
            # Group by dataset and calculate mean
            dataset_metrics = df.groupby('dataset').agg(
                WER=('wer', 'mean'),
                RFx=('rfx', 'mean')
            ).reset_index()

            # Calculate overall average
            avg_wer = df['wer'].mean()
            avg_rfx = df['rfx'].mean()

            # Create a summary DataFrame for display
            summary_df = pd.DataFrame([{'Dataset': 'Average', 'WER': f"{avg_wer:.2%}", 'RFx': f"{avg_rfx:.2f}"}])
            
            # Format dataset-specific metrics
            formatted_dataset_metrics = pd.DataFrame({
                'Dataset': dataset_metrics['dataset'],
                'WER': dataset_metrics['WER'].map('{:.2%}'.format),
                'RFx': dataset_metrics['RFx'].map('{:.2f}'.format)
            })

            # Combine average and dataset metrics
            display_df = pd.concat([summary_df, formatted_dataset_metrics], ignore_index=True)
            
            with results_placeholder.container():
                st.dataframe(display_df, use_container_width=True, hide_index=True)

    else:
        results_placeholder.info("Waiting for benchmark results...")

    # Sleep for a few seconds before checking for new results again
    time.sleep(2)
