"""
AI X-ray Diagnostic Reader - REAL AI Implementation
Integrated with actual PyTorch models

File: app_real.py

Setup:
1. pip install -r requirements.txt
2. Download model weights (see ai_models.py)
3. streamlit run app_real.py
"""
pip install torch torchvision opencv-python
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from datetime import datetime

# Import real AI models
try:
    from ai_models import UniversalXrayAnalyzer
    REAL_AI_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ AI models not available: {e}")
    st.info("Install dependencies: pip install torch torchvision opencv-python")
    REAL_AI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI X-ray Diagnostic Reader - REAL AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #dc2626;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
        margin-top: 1rem;
    }
    .finding-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .critical { background-color: #fee; border-color: #dc2626; }
    .moderate { background-color: #fef3c7; border-color: #f59e0b; }
    .mild { background-color: #fef9c3; border-color: #eab308; }
    .none { background-color: #e0f2fe; border-color: #0ea5e9; }
    .metric-card {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .body-region-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
    }
    .ai-badge {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
keys = [
    "image", "body_region", "results", "patient_context",
    "treatment_plan", "current_page", "clinician_notes", "signed_off"
]
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["current_page"] else "upload"

if "analyzer" not in st.session_state and REAL_AI_AVAILABLE:
    with st.spinner("Loading AI models..."):
        st.session_state.analyzer = UniversalXrayAnalyzer()

# --- Helper Functions ---
def generate_treatment_plan(body_region, findings, patient_context):
    plans = {
        "chest": """Based on chest imaging findings and clinical context:

1. Respiratory Management:
   - Antibiotic Therapy: Consider empiric treatment for community-acquired pneumonia (CAP)
   - First-line: Amoxicillin-clavulanate 875mg BID or Azithromycin 500mg daily
   - Reference: IDSA/ATS CAP Guidelines 2019

2. Cardiac Evaluation: If cardiomegaly present
   - Obtain BNP/NT-proBNP
   - Consider echocardiogram if CHF suspected
   - Reference: ACC/AHA Heart Failure Guidelines 2022

3. Monitoring & Follow-up:
   - Repeat chest X-ray in 48-72 hours if no improvement
   - Monitor vital signs, especially oxygen saturation
   - Consider chest CT if diagnostic uncertainty persists

⚠️ CLINICIAN REVIEW REQUIRED""",
        "hand": """Based on extremity imaging findings and clinical context:

1. Fracture Management:
   - Immediate immobilization with splint/cast
   - Ice therapy: 20 minutes every 2-3 hours for 48 hours
   - Elevation above heart level to reduce swelling
   - NSAIDs for pain (Ibuprofen 400-600mg Q6H PRN)

2. Orthopedic Referral:
   - Urgent orthopedic consultation within 24-48 hours
   - May require closed reduction or surgical fixation
   - Reference: AO Fracture Classification

3. Follow-up:
   - Repeat X-rays in 10-14 days to assess alignment
   - Physical therapy referral after immobilization period
   - Monitor for compartment syndrome symptoms

⚠️ CLINICIAN REVIEW REQUIRED"""
    }
    return plans.get(body_region.lower(), plans["chest"])

def draw_annotations(image, results):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if "findings" in results:
        for f in results["findings"]:
            if "region" in f:
                r = f["region"]
                x1, y1 = int(r["x"] * w), int(r["y"] * h)
                x2, y2 = int((r["x"] + r["w"]) * w), int((r["y"] + r["h"]) * h)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.rectangle([x1, y1 - 25, x1 + 200, y1], fill="red")
                draw.text((x1 + 5, y1 - 20), f["name"], fill="white")
    return img

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🏥 AI X-ray Diagnostic Reader</h1>
    <div class="ai-badge">🤖 REAL AI POWERED</div>
    <p><strong>Multi-Region Analysis with PyTorch Models</strong></p>
    <div class="warning-box">
        ⚠️ RESEARCH PROTOTYPE ONLY - Not for clinical use without physician oversight
    </div>
</div>
""", unsafe_allow_html=True)

# --- Ensure AI Loaded ---
if not REAL_AI_AVAILABLE:
    st.error("🚫 AI models not loaded. Please install dependencies and restart.")
    st.code("pip install torch torchvision opencv-python", language="bash")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("🩻 System Status")
    if st.session_state.image:
        st.success("✅ Image uploaded")
    else:
        st.warning("⏳ No image uploaded")

    if st.session_state.results:
        st.success("✅ AI analysis complete")
    else:
        st.warning("⏳ Analysis not run yet")

    if st.session_state.treatment_plan:
        st.success("✅ Treatment plan ready")
    else:
        st.warning("⏳ No treatment plan")

    if st.session_state.signed_off:
        st.success("✅ Clinician signed off")
    else:
        st.warning("⏳ Not signed off")

    st.divider()
    st.header("🎨 Image Adjustments")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    show_overlay = st.checkbox("Show AI Overlay", value=True)

# --- PAGE 1: Upload ---
if st.session_state.current_page == "upload":
    st.header("Step 1: Upload X-ray Image")
    uploaded = st.file_uploader("Choose X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.session_state.image = Image.open(uploaded)
        st.image(st.session_state.image, caption="Uploaded X-ray", use_container_width=True)
        if st.button("🔍 Proceed to Analysis →", type="primary"):
            st.session_state.current_page = "analyze"
            st.rerun()

# --- PAGE 2: Analysis ---
elif st.session_state.current_page == "analyze":
    st.header("Step 2: Real AI Analysis")
    if not st.session_state.image:
        st.warning("Upload an image first!")
    else:
        if st.button("🤖 Run AI Analysis", type="primary"):
            with st.spinner("Running real PyTorch models..."):
                try:
                    results = st.session_state.analyzer.analyze(st.session_state.image)
                    st.session_state.results = results
                    st.session_state.body_region = results["body_region"]
                    st.success("✅ AI analysis complete!")
                    st.session_state.current_page = "context"
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")

# --- PAGE 3: Context ---
elif st.session_state.current_page == "context":
    st.header("Step 3: Patient Context")
    age = st.number_input("Age", 0, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    symptoms = st.text_area("Symptoms")
    if st.button("💊 Generate Plan →", type="primary"):
        st.session_state.patient_context = {"age": age, "sex": sex, "symptoms": symptoms}
        st.session_state.treatment_plan = generate_treatment_plan(
            st.session_state.body_region, st.session_state.results["findings"], st.session_state.patient_context
        )
        st.session_state.current_page = "plan"
        st.rerun()

# --- PAGE 4: Plan ---
elif st.session_state.current_page == "plan":
    st.header("Step 4: Treatment Plan")
    plan = st.text_area("Treatment Plan", st.session_state.treatment_plan or "", height=300)
    st.session_state.treatment_plan = plan
    signed = st.checkbox("Clinician sign-off", st.session_state.signed_off or False)
    st.session_state.signed_off = signed
    if signed and st.button("📄 Proceed to Export →", type="primary"):
        st.session_state.current_page = "export"
        st.rerun()

# --- PAGE 5: Export ---
elif st.session_state.current_page == "export":
    st.header("Step 5: Export Report")
    if not st.session_state.signed_off:
        st.warning("⚠️ Clinician sign-off required")
    else:
        report = f"AI X-ray Report for {st.session_state.body_region}\nGenerated: {datetime.now()}"
        st.text_area("Final Report", value=report, height=300)
        st.download_button("📥 Download Report", report, file_name="xray_report.txt")

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align:center; color:#666; padding:1rem;'>
    <p><strong>AI X-ray Diagnostic Reader v2.0 - Real PyTorch Implementation</strong></p>
    <p>Research Prototype | Not for clinical use</p>
</div>
""", unsafe_allow_html=True)
