# src/app.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
import logging

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Exo-Detector Mission Control",
    page_icon="🛰️"
)

# --- Path Resolution ---
try:
    APP_DIR = Path(__file__).parent
    PROJECT_ROOT = APP_DIR.parent
    DATA_DIR = PROJECT_ROOT / "data"
    ASSETS_DIR = APP_DIR / "assets"
except NameError:
    DATA_DIR = Path("data")
    ASSETS_DIR = Path("src/assets")

RESULTS_DIR = DATA_DIR / "results"
TRANSIT_WINDOWS_DIR = DATA_DIR / "transit_windows"

# --- Helper Functions ---
def load_css(file_path):
    """Loads a CSS file and applies its styles."""
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            logging.info("Custom CSS loaded successfully.")

def load_json_file(filename):
    """Loads a JSON file from the results directory."""
    filepath = RESULTS_DIR / filename
    return json.load(open(filepath)) if filepath.exists() else None

def create_metric_card(label, value, icon):
    """Creates a styled metric card using markdown."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="status-icon">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Main App ---
logging.basicConfig(level=logging.INFO)

# Apply the custom CSS
load_css(ASSETS_DIR / "style.css")

st.title("🛰️ Exo-Detector Mission Control")
st.markdown("An automated AI pipeline for discovering exoplanet candidates in TESS data.")

# --- Load All Results ---
phase1 = load_json_file("phase1_results.json")
phase2 = load_json_file("phase2_results.json")
phase3 = load_json_file("phase3_results.json")
candidates = load_json_file("final_ranked_candidates.json")

# --- Status Overview Section ---
st.header("Pipeline Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    # *** DEFINITIVE FIX ***
    # This correctly looks inside the 'ingestion' sub-folder within the Phase 1 results.
    ingestion_data = phase1.get('ingestion', {}) if phase1 else {}
    value = ingestion_data.get('real_samples', 'N/A')
    create_metric_card("Real Light Curves", value, "📥")

with col2:
    # This was already correct, but we'll make it more robust.
    value = phase2.get('synthetic_samples_generated', 'N/A') if phase2 else 'N/A'
    create_metric_card("Synthetic Samples", value, "✨")

with col3:
    value = phase3.get('num_test_samples', 'N/A') if phase3 else 'N/A'
    create_metric_card("Windows Analyzed", value, "🔬")

with col4:
    value = len(candidates) if candidates else 'N/A'
    create_metric_card("Top Candidates", value, "🏆")

# --- Candidate Results Section (No changes needed here) ---
st.header("🏆 Top Exoplanet Candidates")
if candidates:
    candidates_df = pd.DataFrame(candidates)
    st.dataframe(candidates_df[['candidate_id', 'anomaly_score', 'is_known_transit']], use_container_width=True)
else:
    st.warning("No candidate data found.")
