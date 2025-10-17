import streamlit as st
import google.generativeai as genai
import openai
from xai_sdk import Client as GrokClient
from xai_sdk.chat import user as grok_user
import os
import io
import yaml
import traceback
from PyPDF2 import PdfReader, PdfWriter
import pytesseract
from pdf2image import convert_from_bytes
import base64
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="🤖 Agentic PDF Processing System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Configurations ---
THEMES = {
    "Blue Sky": {"primary": "#87CEEB", "secondary": "#4682B4", "background": "#F0F8FF", "text": "#1E3A5F", "accent": "#FFD700"},
    "Snow White": {"primary": "#FFFFFF", "secondary": "#E8E8E8", "background": "#F5F5F5", "text": "#2C3E50", "accent": "#3498DB"},
    "Deep Ocean": {"primary": "#006994", "secondary": "#003049", "background": "#001219", "text": "#E0FBFC", "accent": "#F77F00"},
    "Sparkling Galaxy": {"primary": "#8B5CF6", "secondary": "#EC4899", "background": "#1E1B4B", "text": "#F3E8FF", "accent": "#FCD34D"},
    "Alp Forest": {"primary": "#2D5016", "secondary": "#4A7C59", "background": "#E8F5E9", "text": "#1B5E20", "accent": "#FF6F00"},
    "Flora": {"primary": "#E91E63", "secondary": "#9C27B0", "background": "#FCE4EC", "text": "#880E4F", "accent": "#00BCD4"},
    "Ferrari": {"primary": "#DC0000", "secondary": "#8B0000", "background": "#FFF5F5", "text": "#1A0000", "accent": "#FFD700"},
    "Fendi Casa": {"primary": "#D4AF37", "secondary": "#8B7355", "background": "#F5F5DC", "text": "#3E2723", "accent": "#C9A961"}
}

# --- Model Definitions ---
MODEL_OPTIONS = {
    "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Grok": ["grok-1.5-flash", "grok-1.5"]
}

# --- Initialize Session State ---
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {"Gemini": None, "OpenAI": None, "Grok": None}
if 'agents_to_run' not in st.session_state:
    st.session_state.agents_to_run = []
if 'execution_index' not in st.session_state:
    st.session_state.execution_index = 0
if 'results' not in st.session_state:
    st.session_state.results = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False


# --- Custom CSS Function ---
def apply_theme(theme_name):
    theme = THEMES[theme_name]
    css = f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, {theme['background']} 0%, {theme['secondary']}15 100%);
        }}
        .main-header {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            padding: 2rem; border-radius: 15px; text-align: center; color: white;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin-bottom: 2rem; animation: slideIn 0.5s ease-out;
        }}
        .card {{
            background: white; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 1rem 0;
            border-left: 4px solid {theme['accent']}; transition: transform 0.3s ease;
        }}
        .card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.15); }}
        h1, h2, h3 {{ color: {theme['text']}; }}
        .stButton>button {{
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            color: white; border: none; padding: 0.75rem 2rem; border-radius: 25px;
            font-weight: bold; transition: all 0.3s ease;
        }}
        .stButton>button:hover {{ transform: scale(1.05); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        @keyframes slideIn {{ from {{ opacity: 0; transform: translateY(-20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Helper Functions ---
def get_api_key(provider):
    """Checks env, secrets, and session state for API key. Prompts user if not found."""
    key_env_map = {"Gemini": "GOOGLE_API_KEY", "OpenAI": "OPENAI_API_KEY", "Grok": "XAI_API_KEY"}
    key = os.getenv(key_env_map[provider]) or st.secrets.get(key_env_map[provider])
    if key:
        st.session_state.api_keys[provider] = key
        return key

    if st.session_state.api_keys.get(provider):
        return st.session_state.api_keys[provider]

    with st.sidebar:
        st.warning(f"⚠️ {provider} API key not found.")
        user_key = st.text_input(
            f"Enter your {provider} API Key:",
            type="password",
            key=f"{provider}_api_key_input"
        )
        if user_key:
            st.session_state.api_keys[provider] = user_key
            st.rerun()
    return None

def extract_text_from_pdf(file_bytes, use_ocr=False, pages_to_trim=None):
    """Extracts text from PDF, with optional OCR and page trimming."""
    try:
        if pages_to_trim:
            reader = PdfReader(io.BytesIO(file_bytes))
            writer = PdfWriter()
            start, end = pages_to_trim
            for i in range(start - 1, end):
                writer.add_page(reader.pages[i])
            trimmed_pdf_bytes = io.BytesIO()
            writer.write(trimmed_pdf_bytes)
            file_bytes = trimmed_pdf_bytes.getvalue()

        if use_ocr:
            images = convert_from_bytes(file_bytes)
            full_text = ""
            progress_bar = st.progress(0, text="Performing OCR...")
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                full_text += f"\n--- Page {i+1} ---\n{text}"
                progress_bar.progress((i + 1) / len(images))
            progress_bar.empty()
            return full_text
        else:
            reader = PdfReader(io.BytesIO(file_bytes))
            return "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

@st.cache_data
def load_agents_config():
    """Loads agent configurations from agents.yaml."""
    try:
        with open("agents.yaml", 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("agents.yaml not found. Please create it.")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing agents.yaml: {e}")
        return {}

def get_llm_client(api_choice, api_key):
    """Initializes and returns the appropriate LLM client with the provided key."""
    if not api_key:
        st.error(f"{api_choice} API key is missing.")
        return None
    try:
        if api_choice == "Gemini":
            genai.configure(api_key=api_key)
            return genai
        elif api_choice == "OpenAI":
            return openai.OpenAI(api_key=api_key)
        elif api_choice == "Grok":
            return GrokClient(api_key=api_key, timeout=3600)
    except Exception as e:
        st.error(f"Error initializing {api_choice} client: {e}")
        return None

def execute_agent(agent_config, input_text, api_key):
    """Executes a single agent with its full configuration and input."""
    client = get_llm_client(agent_config['api'], api_key)
    if not client:
        return f"Could not initialize {agent_config['api']} client."

    prompt = agent_config['prompt'].format(input_text=input_text)
    model = agent_config['model']

    try:
        with st.spinner(f"🤖 {agent_config['name']} is thinking..."):
            if agent_config['api'] == "Gemini":
                model_instance = client.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
                return response.text
            elif agent_config['api'] == "OpenAI":
                params = agent_config.get('parameters', {})
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )
                return response.choices[0].message.content
            elif agent_config['api'] == "Grok":
                chat = client.chat.create(model=model)
                chat.append(grok_user(prompt))
                return chat.sample().content
    except Exception as e:
        error_message = f"Error executing agent '{agent_config['name']}': {e}"
        st.error(error_message)
        traceback.print_exc()
        return error_message

# --- Main Application ---
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎨 Choose Your Theme")
        selected_theme = st.selectbox("Select Theme", list(THEMES.keys()), index=2)
        st.markdown("---")
        st.markdown("### ⚙️ Global Configuration")
        api_choice = st.selectbox("Select API Provider", list(MODEL_OPTIONS.keys()))
        model_choice = st.selectbox("Select Model", MODEL_OPTIONS[api_choice])
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("This intelligent system processes PDFs using a configurable, sequential chain of AI agents.")

    apply_theme(selected_theme)
    api_key = get_api_key(api_choice)

    # --- Header ---
    st.markdown('<div class="main-header"><h1>🤖 Agentic PDF Processing System</h1>'
                '<p>Transform documents with a customizable, multi-agent AI workflow</p></div>', unsafe_allow_html=True)

    # --- Step 1: PDF Upload & Processing ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Step 1: Upload & Process Your PDF")
    uploaded_file = st.file_uploader("Drop your PDF here", type=['pdf'], help="Upload a PDF for analysis")

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            total_pages = len(reader.pages)
            use_page_range = st.checkbox(f"Process specific pages (Total: {total_pages})")
            if use_page_range:
                start_page, end_page = st.slider("Select page range", 1, total_pages, (1, total_pages))
                pages_to_trim = (int(start_page), int(end_page))
            else:
                pages_to_trim = None
        with col2:
            use_ocr = st.checkbox("Use OCR for scanned PDFs", help="Slower but necessary for image-based text")

        if st.button("📄 Process PDF", use_container_width=True):
            with st.spinner("Analyzing document..."):
                file_bytes = uploaded_file.read()
                st.session_state.extracted_text = extract_text_from_pdf(file_bytes, use_ocr, pages_to_trim)
                if st.session_state.extracted_text.strip():
                    st.session_state.pdf_processed = True
                    st.success("PDF processed successfully! Configure your agents below.")
                else:
                    st.error("No text could be extracted. Try enabling OCR if it's a scanned document.")
                    st.session_state.pdf_processed = False
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 2: Agent Configuration ---
    if st.session_state.pdf_processed:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Step 2: Configure Your AI Agents")
        all_agents_config = load_agents_config()
        if not all_agents_config or 'agents' not in all_agents_config:
            st.error("Could not load agent configurations from agents.yaml.")
            st.stop()

        all_agents = all_agents_config['agents']
        agent_names = [agent['name'] for agent in all_agents]

        num_agents = st.slider("How many agents do you want to run?", 1, 10, 1)

        st.session_state.agents_to_run = []
        for i in range(num_agents):
            st.markdown(f"---")
            st.markdown(f"#### Agent {i+1}")
            selected_agent_name = st.selectbox(f"Select Agent {i+1}", agent_names, key=f"agent_select_{i}", index=i % len(agent_names))
            
            # Find the full agent config
            agent_template = next((agent for agent in all_agents if agent['name'] == selected_agent_name), None)
            
            if agent_template:
                # Create a customizable copy
                custom_agent = agent_template.copy()
                custom_agent['api'] = api_choice
                custom_agent['model'] = model_choice

                with st.expander("Customize Prompt & Parameters"):
                    custom_agent['prompt'] = st.text_area("Agent Prompt", value=agent_template['prompt'], height=200, key=f"prompt_{i}")
                    
                    # Handle parameters, defaulting to empty dict
                    params = agent_template.get('parameters', {})
                    try:
                        modified_params_str = st.text_area("Model Parameters (JSON format)", value=json.dumps(params), key=f"params_{i}")
                        custom_agent['parameters'] = json.loads(modified_params_str)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in parameters. Please correct it.")
                        # Use default params if user input is invalid
                        custom_agent['parameters'] = params

                st.session_state.agents_to_run.append(custom_agent)

        if st.button("✅ Confirm Agent Configuration", use_container_width=True):
            st.session_state.execution_index = 0
            st.session_state.results = []
            st.success(f"{len(st.session_state.agents_to_run)} agents are configured and ready to execute.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 3: Sequential Agent Execution ---
    if st.session_state.agents_to_run and st.session_state.execution_index < len(st.session_state.agents_to_run):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Step 3: Execute Agents")
        
        current_agent_index = st.session_state.execution_index
        current_agent = st.session_state.agents_to_run[current_agent_index]

        st.info(f"Ready to execute Agent {current_agent_index + 1}/{len(st.session_state.agents_to_run)}: **{current_agent['name']}**")
        
        if st.button(f"🚀 Execute Agent: {current_agent['name']}", use_container_width=True):
            if not api_key:
                st.error(f"Cannot execute agent. Please provide the {api_choice} API key in the sidebar.")
            else:
                result = execute_agent(current_agent, st.session_state.extracted_text, api_key)
                st.session_state.results.append({"name": current_agent['name'], "result": result})
                st.session_state.execution_index += 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Display Results ---
    if st.session_state.results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Agent Results")
        for res in st.session_state.results:
            with st.expander(f"✅ Result from: {res['name']}", expanded=True):
                st.markdown(res['result'])
                st.download_button(
                    f"💾 Download Result",
                    res['result'],
                    file_name=f"{res['name'].replace(' ', '_')}_result.txt"
                )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.agents_to_run and st.session_state.execution_index == len(st.session_state.agents_to_run):
        st.success("🎉 All agents have completed their tasks!")


    # --- Follow-up Questions Section ---
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 What's Next?")
    questions = [
        "Would you like to add **batch processing** for multiple PDFs simultaneously?",
        "Should I implement an **AI-powered summarization** with adjustable length controls?",
        "Would you like to add **export options** (e.g., Markdown, JSON, DOCX) for all processed results combined?",
        "Should I create **custom agent workflows** where you can chain multiple agents and pass outputs as inputs?",
        "Would you like **real-time collaboration features** for multi-user document processing?"
    ]
    for i, question in enumerate(questions, 1):
        st.markdown(f"{i}. {question}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
