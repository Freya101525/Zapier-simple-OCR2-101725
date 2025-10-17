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
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"],
    "Grok": ["grok-4-fast-reasoning", "grok-3-mini"]
}

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        'api_keys': {"Gemini": None, "OpenAI": None, "Grok": None},
        'agents_to_run': [],
        'execution_index': 0,
        'results': [],
        'extracted_text': "",
        'editable_text': "",
        'pdf_processed': False,
        'text_confirmed': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

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
    key_env_map = {"Gemini": "GEMINI_API_KEY", "OpenAI": "OPENAI_API_KEY", "Grok": "XAI_API_KEY"}
    key = os.getenv(key_env_map[provider]) or st.secrets.get(key_env_map[provider])
    if key:
        st.session_state.api_keys[provider] = key
        return key
    if st.session_state.api_keys.get(provider):
        return st.session_state.api_keys[provider]
    with st.sidebar:
        st.warning(f"⚠️ {provider} API key not found.")
        user_key = st.text_input(f"Enter your {provider} API Key:", type="password", key=f"{provider}_api_key_input")
        if user_key:
            st.session_state.api_keys[provider] = user_key
            st.rerun()
    return None

def extract_text_from_pdf(file_bytes, use_ocr=False, pages_to_trim=None):
    try:
        if pages_to_trim:
            reader = PdfReader(io.BytesIO(file_bytes))
            writer = PdfWriter()
            start, end = pages_to_trim
            for i in range(start - 1, min(end, len(reader.pages))):
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
    try:
        with open("agents.yaml", 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading or parsing agents.yaml: {e}")
        return {}

def get_llm_client(api_choice, api_key):
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
    client = get_llm_client(agent_config['api'], api_key)
    if not client: return f"Could not initialize {agent_config['api']} client."
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
                response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **params)
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
    with st.sidebar:
        st.markdown("### 🎨 Choose Your Theme")
        selected_theme = st.selectbox("Select Theme", list(THEMES.keys()), index=2)
        st.markdown("---")
        st.markdown("### ⚙️ Global Configuration")
        api_choice = st.selectbox("Select API Provider", list(MODEL_OPTIONS.keys()))
        model_choice = st.selectbox("Select Model", MODEL_OPTIONS[api_choice])
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("A multi-step, agentic system to process, edit, and analyze PDF documents using AI.")

    apply_theme(selected_theme)
    api_key = get_api_key(api_choice)

    st.markdown('<div class="main-header"><h1>🤖 Agentic PDF Processing System</h1>'
                '<p>Transform documents with a customizable, multi-agent AI workflow</p></div>', unsafe_allow_html=True)

    # --- Step 1: PDF Upload & Initial Processing ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Step 1: Upload & Process Your PDF")
    uploaded_file = st.file_uploader("Drop your PDF here", type=['pdf'])
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            total_pages = len(reader.pages)
            use_page_range = st.checkbox(f"Process specific pages (Total: {total_pages})", key="page_range_check")
            pages_to_trim = st.slider("Select page range", 1, total_pages, (1, total_pages), key="page_slider") if use_page_range else None
        with col2:
            use_ocr = st.checkbox("Use OCR for scanned PDFs", help="Slower but necessary for image-based text")
        if st.button("📄 Process PDF", use_container_width=True):
            # Reset state for a new file
            st.session_state.pdf_processed = False
            st.session_state.text_confirmed = False
            st.session_state.agents_to_run = []
            st.session_state.results = []
            with st.spinner("Analyzing document..."):
                file_bytes = uploaded_file.read()
                extracted = extract_text_from_pdf(file_bytes, use_ocr, pages_to_trim)
                if extracted.strip():
                    st.session_state.extracted_text = extracted
                    st.session_state.editable_text = extracted
                    st.session_state.pdf_processed = True
                    st.success("PDF processed! Please review the extracted text below.")
                else:
                    st.error("No text extracted. Try enabling OCR for scanned documents.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 2: Review & Edit Extracted Text ---
    if st.session_state.pdf_processed and not st.session_state.text_confirmed:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Step 2: Review & Edit Extracted Text")
        st.info("Correct any OCR errors or modify the text below before sending it to the AI agents. The agents will only see the final version from this text box.")
        edited_text = st.text_area("Editable Text Content", value=st.session_state.editable_text, height=400, key="editor")
        if st.button("✅ Confirm Text for Analysis", use_container_width=True):
            st.session_state.editable_text = edited_text
            st.session_state.text_confirmed = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 3: Agent Configuration ---
    if st.session_state.text_confirmed:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Step 3: Configure Your AI Agents")
        all_agents_config = load_agents_config()
        if not all_agents_config or 'agents' not in all_agents_config:
            st.stop()
        all_agents = all_agents_config['agents']
        agent_names = [agent['name'] for agent in all_agents]
        num_agents = st.slider("How many agents to run in sequence?", 1, 10, 1)
        st.session_state.agents_to_run = []
        for i in range(num_agents):
            st.markdown(f"---")
            st.markdown(f"#### Agent {i+1}")
            selected_agent_name = st.selectbox(f"Select Agent {i+1}", agent_names, key=f"agent_{i}", index=i % len(agent_names))
            agent_template = next((a for a in all_agents if a['name'] == selected_agent_name), None)
            if agent_template:
                custom_agent = agent_template.copy()
                custom_agent['api'] = api_choice
                custom_agent['model'] = model_choice
                with st.expander("Customize Prompt & Parameters"):
                    custom_agent['prompt'] = st.text_area("Agent Prompt", value=agent_template['prompt'], height=200, key=f"prompt_{i}")
                    params = agent_template.get('parameters', {})
                    try:
                        modified_params_str = st.text_area("Model Parameters (JSON)", value=json.dumps(params), key=f"params_{i}")
                        custom_agent['parameters'] = json.loads(modified_params_str)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in parameters. Using defaults.")
                        custom_agent['parameters'] = params
                st.session_state.agents_to_run.append(custom_agent)
        if st.button("✅ Confirm Agent Configuration", use_container_width=True):
            st.session_state.execution_index = 0
            st.session_state.results = []
            st.success(f"{len(st.session_state.agents_to_run)} agents are ready to execute.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 4: Sequential Agent Execution ---
    if st.session_state.agents_to_run and st.session_state.execution_index < len(st.session_state.agents_to_run):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Step 4: Execute Agents")
        idx = st.session_state.execution_index
        agent = st.session_state.agents_to_run[idx]
        st.info(f"Ready to execute Agent {idx + 1}/{len(st.session_state.agents_to_run)}: **{agent['name']}**")
        if st.button(f"🚀 Execute Agent: {agent['name']}", use_container_width=True):
            if not api_key:
                st.error(f"Please provide the {api_choice} API key in the sidebar.")
            else:
                result = execute_agent(agent, st.session_state.editable_text, api_key) # Use edited text
                st.session_state.results.append({"name": agent['name'], "result": result})
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
                st.download_button(f"💾 Download Result", res['result'], file_name=f"{res['name'].replace(' ', '_')}_result.txt")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.session_state.execution_index == len(st.session_state.agents_to_run) and st.session_state.agents_to_run:
            st.success("🎉 All agents have completed their tasks!")

    # --- Follow-up Questions ---
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💡 What's Next?")
    questions = [
        "Would you like to add a **'Reset' button** to start over with a new PDF?",
        "Should I implement a feature to **save and load agent configurations** as templates?",
        "Would you like an option for agents to **run in parallel** instead of sequentially?",
        "Should I add a **'diff' view** to show the changes between the original and edited text?",
        "Would you like to add a final step to **synthesize all agent results** into a single report?"
    ]
    for i, question in enumerate(questions, 1):
        st.markdown(f"{i}. {question}")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
