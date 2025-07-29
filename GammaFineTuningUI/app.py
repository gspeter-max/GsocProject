import streamlit as st
import pandas as pd
import torch
import logging
import os
import shutil
import subprocess
from threading import Thread
import time
import inspect
from queue import Queue

# Import your backend modules
from backend.GlobalConfig import GetIt
from backend.DatasetUpLoading import UploadDataset
from backend.ModelingAndTuning import ModelLoadingAndTuning, map_function, tokenizer
from backend.GetModel import ConvertModel

# For live plotting
from tensorboard.backend.event_processing import event_accumulator
import plotly.graph_objects as go

# =====================================================================================
# 1. App Configuration and Styling
# =====================================================================================

st.set_page_config(
    page_title="Aurora AI Fine-Tuner ✨",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a beautiful, modern look ---
st.markdown("""
<style>
    /* Import a Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

    /* Main app styling */
    html, body, [class*="st-"] {
        font-family: 'Roboto Mono', monospace;
    }

    .stApp {
        background-color: #000000;
        background-image: linear-gradient(180deg, #0a0a1a 0%, #000000 100%);
        color: #E0E0E0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(10, 10, 26, 0.8);
        border-right: 1px solid #7B00B1;
    }

    /* Title and Header styling */
    h1, h2, h3 {
        color: #9D4EDD; /* A vibrant purple */
        text-shadow: 0 0 5px #9D4EDD;
    }

    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #7B00B1;
        background-image: linear-gradient(to right, #7B00B1, #9D4EDD);
        color: white;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px #7B00B1;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #9D4EDD;
        transform: scale(1.02);
    }
    .stButton>button:disabled {
        background: #555;
        color: #888;
        border-color: #555;
        box-shadow: none;
    }

    /* Expander styling */
    .st-expander {
        border-radius: 10px;
        border: 1px solid #7B00B1;
        background-color: rgba(20, 20, 40, 0.5);
    }
    .st-expander header {
        color: #C77DFF;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        border-bottom: 2px solid #555;
        color: #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #9D4EDD;
        color: #9D4EDD;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# =====================================================================================
# 2. Session State Initialization
# =====================================================================================

def init_session_state():
    if 'config' not in st.session_state:
        # Initialize config from GlobalConfig defaults
        default_global = GetIt()
        default_training = GetIt.GetTrainingArguments()
        default_peft = GetIt.GetPeftConfig()
        default_tokenization = GetIt.GetTokenizationConfig()
        
        st.session_state.config = {
            "global": {k: v for k, v in default_global.__dict__.items()},
            "training": default_training,
            "peft": default_peft,
            "tokenization": default_tokenization
        }
    
    if 'training_status' not in st.session_state:
        st.session_state.training_status = "Not Started" # Can be "Not Started", "Running", "Completed", "Error"
    
    if 'training_thread' not in st.session_state:
        st.session_state.training_thread = None
    
    if 'log_queue' not in st.session_state:
        st.session_state.log_queue = Queue()

init_session_state()

# =====================================================================================
# 3. Backend Logic Integration
# =====================================================================================

# Custom handler to redirect logging to a Streamlit queue
class QueueLogHandler(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        self.queue.put(self.format(record))

def training_worker(config):
    try:
        st.session_state.training_status = "Running"
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        handler = QueueLogHandler(st.session_state.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        st.session_state.log_queue.put("--- Initializing Training ---")
        
        # This part needs to be adapted based on how your backend scripts
        # consume the configuration. We assume they can be passed a config dict.
        
        # This is a placeholder for your actual training logic call
        # You would need to refactor ModelingAndTuning.py to accept the config dict
        # For now, we'll simulate it.
        st.session_state.log_queue.put(f"CONFIG: {config}")
        
        # --- Simulating your backend logic ---
        st.session_state.log_queue.put("Loading dataset...")
        time.sleep(2) # Simulate work
        st.session_state.log_queue.put("Tokenizing data...")
        time.sleep(2) # Simulate work
        st.session_state.log_queue.put("Loading model and applying PEFT...")
        time.sleep(3) # Simulate work
        st.session_state.log_queue.put("Starting Trainer...")

        # Create a dummy log file for plotting
        log_dir = config['training']['logging_dir']
        os.makedirs(log_dir, exist_ok=True)
        event_file_path = os.path.join(log_dir, "events.out.tfevents.dummy")
        with open(event_file_path, "w") as f:
            f.write("This is a dummy file for demonstration.")

        for epoch in range(int(config['training']['num_train_epochs'])):
            st.session_state.log_queue.put(f"--- Epoch {epoch + 1} ---")
            for step in range(10): # Simulate steps
                loss = 1.0 / (step + 1)
                st.session_state.log_queue.put(f"Step {step+1}/10 - loss: {loss:.4f}")
                time.sleep(0.5)

        st.session_state.log_queue.put("--- Training Completed Successfully ---")
        st.session_state.training_status = "Completed"

    except Exception as e:
        st.session_state.log_queue.put(f"--- ERROR: {str(e)} ---")
        st.session_state.training_status = "Error"

# =====================================================================================
# 4. UI Rendering Functions
# =====================================================================================

def render_home_page():
    st.title("Welcome to the Aurora AI Fine-Tuner ✨")
    st.markdown("""
    This application provides a powerful and intuitive interface to fine-tune Hugging Face models using PEFT/LoRA.
    
    **Navigate through the sections using the sidebar:**
    - **🏠 Home:** You are here!
    - **🛠️ Configuration:** Set up your dataset, model, and all training hyperparameters.
    - **🚀 Train & Monitor:** Launch the training process and monitor its progress in real-time.
    - **📊 Results & Analysis:** View and analyze the results after training is complete.
    
    Ready to build? Let's get started!
    """, unsafe_allow_html=True)
    st.image("https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/Hugging-Face-on-Google-Cloud.png", caption="Harness the power of Hugging Face with an elegant UI.")


def render_config_page():
    st.header("🛠️ Configuration")
    
    with st.expander("📂 Dataset & Model", expanded=True):
        st.session_state.config['global']['ModelName'] = st.text_input(
            "Model Name", st.session_state.config['global']['ModelName'], help="Base model from Hugging Face.")
        st.session_state.config['global']['QuantizationType4Bit8Bit'] = st.checkbox(
            "Enable 8-bit Quantization", st.session_state.config['global']['QuantizationType4Bit8Bit'])
        
        # Ideally, your UploadDataset class would be used here. For simplicity:
        st.text_input("Dataset Path", "qualifire/grounding-benchmark")
        st.text_input("Dataset Split", "train")

    with st.expander("⚙️ Training Arguments", expanded=False):
        # Dynamically create UI elements from the config dict
        for key, value in st.session_state.config['training'].items():
            if isinstance(value, bool):
                st.session_state.config['training'][key] = st.checkbox(key, value)
            elif isinstance(value, float):
                st.session_state.config['training'][key] = st.number_input(key, value=value, format="%.0e" if "rate" in key else "%.2f")
            elif isinstance(value, int):
                st.session_state.config['training'][key] = st.number_input(key, value=value, step=1)
            elif isinstance(value, str) or value is None:
                st.session_state.config['training'][key] = st.text_input(key, value)

    with st.expander("🔧 PEFT (LoRA) Arguments", expanded=False):
        for key, value in st.session_state.config['peft'].items():
            if isinstance(value, bool):
                st.session_state.config['peft'][key] = st.checkbox(key, value)
            elif isinstance(value, float):
                st.session_state.config['peft'][key] = st.slider(key, 0.0, 1.0, value)
            elif isinstance(value, int):
                st.session_state.config['peft'][key] = st.slider(key, 1, 128, value)
            elif isinstance(value, str):
                st.session_state.config['peft'][key] = st.text_input(key, value)
            elif isinstance(value, list):
                 st.session_state.config['peft'][key] = st.multiselect(key, options=value, default=value)
                 
    st.success("Configuration updated. Proceed to the 'Train & Monitor' page.")


def render_train_page():
    st.header("🚀 Train & Monitor")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Control Panel")
        start_button = st.button("Start Training", disabled=(st.session_state.training_status == "Running"))
        
        if start_button:
            thread = Thread(target=training_worker, args=(st.session_state.config,))
            st.session_state.training_thread = thread
            thread.start()
        
        st.write(f"**Status:** {st.session_state.training_status}")
        
    with col2:
        st.subheader("Live Metrics")
        # In a real scenario, you'd parse real logs here. We'll simulate.
        if st.session_state.training_status == "Running":
            # Create placeholders for live plots
            loss_chart_placeholder = st.empty()
            
            # This loop simulates live updating
            loss_data = []
            for i in range(50):
                loss_data.append(1 / (i + 1) + torch.randn(1).item() * 0.1)
                fig = go.Figure(data=go.Scatter(y=loss_data, mode='lines', line=dict(color='#9D4EDD')))
                fig.update_layout(title="Live Training Loss", xaxis_title="Step", yaxis_title="Loss", template="plotly_dark")
                loss_chart_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.2)

    st.subheader("Live Console Logs")
    log_placeholder = st.empty()
    log_container = log_placeholder.container()
    
    current_logs = []
    def update_logs():
        while not st.session_state.log_queue.empty():
            current_logs.append(st.session_state.log_queue.get_nowait())
        log_container.code("\n".join(current_logs), language='log')

    # Keep updating logs while the thread is alive
    if st.session_state.training_status == "Running":
        update_logs()
    
def render_results_page():
    st.header("📊 Results & Analysis")
    if st.session_state.training_status == "Completed":
        st.success("Training finished successfully!")
        st.balloons()
        
        # Add download buttons for model/logs
        st.subheader("Download Artifacts")
        # In a real app, you would zip the model directory before offering a download
        st.button("Download Trained Model (Not Implemented)")
        st.button("Download Logs (Not Implemented)")
        
    elif st.session_state.training_status == "Error":
        st.error("Training failed. Check the logs on the 'Train & Monitor' page for details.")
    else:
        st.info("Training has not yet completed. Please start a training run and wait for it to finish.")

# =====================================================================================
# 5. Main App Router
# =====================================================================================

PAGES = {
    "🏠 Home": render_home_page,
    "🛠️ Configuration": render_config_page,
    "🚀 Train & Monitor": render_train_page,
    "📊 Results & Analysis": render_results_page,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()

st.sidebar.info(f"Current Status: {st.session_state.training_status}")
