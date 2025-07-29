import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import torch
import logging
import os
import time
from threading import Thread
from queue import Queue
import re

# Backend Imports using relative paths from within the package
from backend.GlobalConfig import GetIt
from backend.ModelingAndTuning import ModelLoadingAndTuning
from backend.DatasetUpLoading import UploadDataset

# For live plotting from log files
import plotly.graph_objects as go
from tensorboard.backend.event_processing import event_accumulator

# =====================================================================================
# 1. APP CONFIGURATION AND STYLING
# =====================================================================================
st.set_page_config(
    page_title="Aurora Tuner ✨ | AI Model Fine-Tuning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&display=swap');

    html, body, [class*="st-"], .st-emotion-cache-10trblm {
        font-family: 'Fira Code', monospace;
    }
    .stApp {
        background-color: #010001;
        color: #E6E6E6;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 3rem 1rem;
    }
    h1, h2, h3 {
        color: #B388FF;
        text-shadow: 0 0 3px #B388FF, 0 0 5px #D1C4E9;
    }
    .stButton>button {
        border-radius: 5px;
        border: 1px solid #7E57C2;
        background: linear-gradient(45deg, #4527A0, #673AB7);
        color: white;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px 0 rgba(126, 87, 194, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(126, 87, 194, 0.6);
    }
    .st-expander {
        border-color: #512DA8;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# 2. SESSION STATE & HELPERS
# =====================================================================================
if 'config' not in st.session_state:
    st.session_state.config = GetIt() # Initialize with defaults
if 'training_status' not in st.session_state:
    st.session_state.training_status = "Not Started"
if 'training_thread' not in st.session_state:
    st.session_state.training_thread = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = Queue()

def get_default_params(func):
    """Helper to get default values from function signatures."""
    return {param.name: param.default for param in inspect.signature(func).parameters.values()}

# =====================================================================================
# 3. TRAINING WORKER & LOGGING
# =====================================================================================
class QueueLogHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
    def emit(self, record):
        self.queue.put(self.format(record))

def training_worker(config_obj):
    try:
        st.session_state.training_status = "Running"
        # Setup logging
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = QueueLogHandler(st.session_state.log_queue)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        
        st.session_state.log_queue.put("--- [INITIALIZING] ---")
        st.session_state.log_queue.put(f"Configuration received. Model: {config_obj.ModelName}")

        # Instantiate and run your backend class
        tuner = ModelLoadingAndTuning(config_obj)
        tuner.LoadItTrainIt()
        
        st.session_state.log_queue.put("--- [SUCCESS] Training has completed successfully! ---")
        st.session_state.training_status = "Completed"

    except Exception as e:
        st.session_state.log_queue.put(f"--- [CRITICAL ERROR] --- \n{str(e)}")
        st.session_state.training_status = "Error"
    finally:
        if logger.handlers:
            logger.removeHandler(handler)

# =====================================================================================
# 4. UI PAGES
# =====================================================================================

def page_configurator():
    st.title("🛠️ Model & Training Configurator")
    st.markdown("Define every hyperparameter for your training run. Changes are saved automatically.")
    
    config = st.session_state.config
    
    with st.expander("📂 **Core Configuration**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            config.ModelName = st.text_input("Model Name", value=config.ModelName, help="Base model from Hugging Face.")
            config.HfToken = st.text_input("Hugging Face Token", type="password", help="Required for private models. Set as a secret in Streamlit for production.")
        with col2:
            config.SaveFormat = st.selectbox("Save Format", [None, 'torch', 'tensorflow', 'gguf'], index=0 if config.SaveFormat is None else [None, 'torch', 'tensorflow', 'gguf'].index(config.SaveFormat))
            config.ModelDir = st.text_input("Save Directory", value=config.ModelDir)

    with st.expander("⚙️ **Training Arguments**"):
        training_args = get_default_params(GetIt.GetTrainingArguments)
        user_training_args = {}
        for key, value in training_args.items():
            if isinstance(value, bool):
                user_training_args[key] = st.checkbox(key, value)
            elif isinstance(value, float):
                user_training_args[key] = st.number_input(key, value=value, format="%.0e" if "rate" in key else "%.2f")
            elif isinstance(value, int):
                user_training_args[key] = st.number_input(key, value=value, step=10)
            else:
                user_training_args[key] = st.text_input(key, value=str(value))
        config.TrainingArguments = GetIt.GetTrainingArguments(**user_training_args)
    
    with st.expander("🔧 **PEFT (LoRA) Arguments**"):
        peft_args = get_default_params(GetIt.GetPeftConfig)
        user_peft_args = {}
        for key, value in peft_args.items():
            if isinstance(value, bool):
                user_peft_args[key] = st.checkbox(key, value, key=f"peft_{key}")
            elif isinstance(value, float):
                user_peft_args[key] = st.slider(key, 0.0, 1.0, value, key=f"peft_{key}")
            elif isinstance(value, int):
                user_peft_args[key] = st.slider(key, 1, 128, value, key=f"peft_{key}")
            else:
                 user_peft_args[key] = st.text_input(key, value, key=f"peft_{key}")
        config.PeftConfig = GetIt.GetPeftConfig(**user_peft_args)

    st.session_state.config = config
    st.success("Configuration is set. Proceed to the **Train & Monitor** page to launch.")

def page_trainer():
    st.title("🚀 Train & Monitor")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Control Panel")
        start_disabled = st.session_state.training_status == "Running"
        if st.button("🚀 Launch Training!", disabled=start_disabled, use_container_width=True):
            # Pass a copy of the finalized config object
            final_config = st.session_state.config
            # Call the __call__ method to get the dictionary representation for the backend
            hyperparameter_dict = final_config(
                TokenizationConfig=final_config.GetTokenizationConfig(), # Assuming defaults are okay for now
                PeftConfig=final_config.PeftConfig,
                TrainingArguments=final_config.TrainingArguments
            )

            # Important: The backend needs to be refactored to accept this dictionary
            # For now, we launch with the object itself
            thread = Thread(target=training_worker, args=(final_config,))
            st.session_state.training_thread = thread
            thread.start()
            st.rerun()

        st.metric("Training Status", st.session_state.training_status)

        if st.session_state.training_status == "Running":
            with st.spinner("Training in progress..."):
                time.sleep(1) # Small delay to allow logs to populate
            st.rerun()

    with col2:
        st.subheader("Live Console")
        log_placeholder = st.empty()
        log_records = []
        while not st.session_state.log_queue.empty():
            log_records.append(st.session_state.log_queue.get())
        
        if log_records:
            st.session_state.full_log = st.session_state.get('full_log', '') + '\n'.join(log_records)
        
        log_placeholder.code(st.session_state.get('full_log', 'Logs will appear here...'), language='log', height=400)

    st.divider()
    st.subheader("📈 Live Training Graphs")
    # This section needs to be connected to the log directory
    st.info("Live graphs will appear here once training starts and log files are generated.")
    # Placeholder for live graph
    graph_placeholder = st.empty()
    # In a real scenario, a separate thread would monitor the log dir and update this plot
    
def page_results():
    st.title("📊 Results & Analysis")
    if st.session_state.training_status == "Completed":
        st.balloons()
        st.success("Training run completed successfully!")
        st.info(f"You can find your saved model in the directory: `{st.session_state.config.ModelDir}` if specified.")
        # Logic to display final metrics would go here
    else:
        st.warning("No completed training run found. Please complete a run on the **Train & Monitor** page.")

# =====================================================================================
# 5. MAIN APP ROUTER
# =====================================================================================

with st.sidebar:
    selected = option_menu(
        menu_title="Aurora Tuner",
        options=["Configurator", "Train & Monitor", "Results"],
        icons=["gear-wide-connected", "rocket-takeoff", "bar-chart-line"],
        menu_icon="robot",
        default_index=0,
        styles={
            "container": {"background-color": "#010001"},
            "icon": {"color": "#D1C4E9", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4527A0"},
            "nav-link-selected": {"background-color": "#512DA8"},
        }
    )

if selected == "Configurator":
    page_configurator()
elif selected == "Train & Monitor":
    page_trainer()
elif selected == "Results":
    page_results()