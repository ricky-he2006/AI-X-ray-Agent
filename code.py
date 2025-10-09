"""
AI X-ray Diagnostic Reader - Streamlit Version
Research Prototype - Requires clinician review

Installation:
pip install streamlit numpy pillow matplotlib

Usage:
streamlit run xray_diagnostic_app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import io
import time

# Page configuration
st.set_page_config(
    page_title="AI X-ray Diagnostic Reader",
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
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'patient_context' not in st.session_state:
    st.session_state.patient_context = {}
if 'treatment_plan' not in st.session_state:
    st.session_state.treatment_plan = ""
if 'step' not in st.session_state:
    st.session_state.step = 1

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI X-ray Diagnostic Reader</h1>
    <p><strong>Doctor-in-the-Loop System v1.0</strong></p>
    <div class="warning-box">
        ⚠️ RESEARCH PROTOTYPE ONLY - Not for clinical use without physician oversight
    </div>
</div>
""", unsafe_allow_html=True)

# Progress indicator
progress_steps = ["Upload", "Analyze", "Context", "Plan", "Export"]
cols = st.columns(len(progress_steps))
for idx, (col, step) in enumerate(zip(cols, progress_steps)):
    with col:
        if idx + 1 < st.session_state.step:
            st.success(f"✅ {step}")
        elif idx + 1 == st.session_state.step:
            st.info(f"▶️ {step}")
        else:
            st.text(f"⏸️ {step}")

st.divider()

# Sidebar
with st.sidebar:
    st.header("📊 Current Status")
    st.write(f"**Step:** {st.session_state.step}/5")
    
    if st.session_state.image:
        st.success("✅ Image uploaded")
    else:
        st.warning("⏳ No image")
    
    if st.session_state.results:
        st.success("✅ Analysis complete")
    else:
        st.warning("⏳ Not analyzed")
    
    st.divider()
    st.header("🎨 Image Adjustments")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    zoom = st.slider("Zoom", 0.5, 3.0, 1.0, 0.25)
    show_overlay = st.checkbox("Show AI Overlay", value=True)

# Mock AI analysis function
def mock_ai_analysis():
    """Simulate AI model inference"""
    return {
        'model_version': 'DenseNet-121-CheXpert-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Cardiomegaly',
                'confidence': 0.78,
                'severity': 'moderate',
                'region': {'x': 0.35, 'y': 0.40, 'w': 0.30, 'h': 0.25},
                'description': 'Enlarged cardiac silhouette with cardiothoracic ratio >0.5'
            },
            {
                'name': 'Pleural Effusion',
                'confidence': 0.65,
                'severity': 'mild',
                'region': {'x': 0.15, 'y': 0.55, 'w': 0.20, 'h': 0.30},
                'description': 'Blunting of costophrenic angle suggestive of fluid'
            },
            {
                'name': 'Lung Opacity',
                'confidence': 0.82,
                'severity': 'moderate',
                'region': {'x': 0.60, 'y': 0.35, 'w': 0.25, 'h': 0.20},
                'description': 'Patchy opacity in right mid-lung zone'
            }
        ],
        'differentials': [
            'Community-acquired pneumonia',
            'Congestive heart failure',
            'Pulmonary edema',
            'Atypical infection'
        ],
        'urgency': 'moderate',
        'auroc': 0.87
    }

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload", "🤖 AI Analysis", "📋 Patient Context", "💊 Treatment Plan", "📄 Export"])

# TAB 1: Upload
with tab1:
    st.header("Step 1: Upload X-ray Image")
    
    uploaded_file = st.file_uploader("Choose an X-ray image", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)
        st.session_state.step = max(st.session_state.step, 2)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Apply adjustments
            img = st.session_state.image.copy()
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
            
            # Display with overlay if results exist
            if st.session_state.results and show_overlay:
                draw = ImageDraw.Draw(img)
                w, h = img.size
                
                for finding in st.session_state.results['findings']:
                    r = finding['region']
                    x1, y1 = int(r['x'] * w), int(r['y'] * h)
                    x2, y2 = int((r['x'] + r['w']) * w), int((r['y'] + r['h']) * h)
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    # Draw label
                    draw.rectangle([x1, y1-25, x1+150, y1], fill='red')
                    draw.text((x1+5, y1-20), finding['name'], fill='white')
            
            st.image(img, caption='X-ray Image', use_container_width=True)
        
        with col2:
            st.metric("Image Size", f"{st.session_state.image.size[0]} x {st.session_state.image.size[1]}")
            st.metric("Mode", st.session_state.image.mode)
            st.metric("Format", uploaded_file.type)
            
            if st.button("🔍 Proceed to Analysis", type="primary", key="proceed_analyze"):
                st.session_state.step = 2
                st.rerun()

# TAB 2: AI Analysis
with tab2:
    st.header("Step 2: AI Analysis")
    
    if st.session_state.image is None:
        st.warning("⚠️ Please upload an image first!")
    else:
        if st.button("🤖 Run AI Analysis", type="primary", key="run_analysis"):
            with st.spinner("Running DenseNet-121 inference..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Preprocessing image...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                status_text.text("Extracting features...")
                progress_bar.progress(50)
                time.sleep(0.8)
                
                status_text.text("Computing predictions...")
                progress_bar.progress(75)
                time.sleep(0.7)
                
                st.session_state.results = mock_ai_analysis()
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                time.sleep(0.3)
                
                st.session_state.step = max(st.session_state.step, 3)
                st.rerun()
        
        if st.session_state.results:
            st.success("✅ Analysis Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", "DenseNet-121")
            with col2:
                st.metric("AUROC", st.session_state.results['auroc'])
            with col3:
                st.metric("Urgency", st.session_state.results['urgency'].upper())
            
            st.divider()
            
            st.subheader("🔍 AI Findings")
            for i, finding in enumerate(st.session_state.results['findings'], 1):
                severity_class = finding['severity']
                st.markdown(f"""
                <div class="finding-box {severity_class}">
                    <strong>{i}. {finding['name']}</strong> 
                    <span style="float: right;">Confidence: {finding['confidence']*100:.0f}%</span>
                    <br>
                    <small><em>{finding['description']}</em></small>
                    <br>
                    <small>Severity: <strong>{finding['severity'].upper()}</strong></small>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            st.subheader("🩺 Differential Diagnoses")
            for i, dx in enumerate(st.session_state.results['differentials'], 1):
                st.write(f"{i}. {dx}")
            
            if st.button("📋 Proceed to Patient Context", type="primary"):
                st.session_state.step = 3
                st.rerun()

# TAB 3: Patient Context
with tab3:
    st.header("Step 3: Patient Clinical Context")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please complete AI analysis first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=0)
            sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        
        with col2:
            symptoms = st.text_area("Chief Complaint / Symptoms", 
                                   placeholder="e.g., Cough, fever for 3 days",
                                   height=100)
        
        st.divider()
        st.subheader("Vital Signs")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.text_input("Temperature (°F)", placeholder="98.6")
            bp = st.text_input("Blood Pressure (mmHg)", placeholder="120/80")
        with col2:
            hr = st.text_input("Heart Rate (bpm)", placeholder="72")
            rr = st.text_input("Respiratory Rate", placeholder="16")
        with col3:
            spo2 = st.text_input("SpO2 (%)", placeholder="98")
        
        if st.button("💾 Save Context & Generate Plan", type="primary", key="save_context"):
            st.session_state.patient_context = {
                'age': age,
                'sex': sex,
                'symptoms': symptoms,
                'smoking_status': smoking_status,
                'vitals': {
                    'temp': temp,
                    'bp': bp,
                    'hr': hr,
                    'rr': rr,
                    'spo2': spo2
                }
            }
            
            # Generate initial treatment plan
            st.session_state.treatment_plan = f"""Based on imaging findings and clinical context:

1. Antibiotic Therapy: Consider empiric treatment for community-acquired pneumonia (CAP)
   - First-line: Amoxicillin-clavulanate 875mg BID or Azithromycin 500mg daily
   - Reference: IDSA/ATS CAP Guidelines 2019

2. Cardiac Evaluation: Given cardiomegaly
   - Obtain BNP/NT-proBNP
   - Consider echocardiogram if CHF suspected
   - Reference: ACC/AHA Heart Failure Guidelines 2022

3. Monitoring & Follow-up:
   - Repeat chest X-ray in 48-72 hours if no improvement
   - Monitor vital signs, especially oxygen saturation
   - Consider chest CT if diagnostic uncertainty persists

⚠️ CLINICIAN REVIEW REQUIRED - This is a draft plan for editing"""
            
            st.session_state.step = max(st.session_state.step, 4)
            st.success("✅ Context saved! Proceeding to treatment plan...")
            time.sleep(1)
            st.rerun()

# TAB 4: Treatment Plan
with tab4:
    st.header("Step 4: Evidence-Based Treatment Plan")
    
    if st.session_state.treatment_plan == "":
        st.warning("⚠️ Please complete patient context first!")
    else:
        st.info("📝 Review and edit the AI-generated treatment plan below. Add your clinical judgment and modifications.")
        
        treatment_plan = st.text_area(
            "Treatment Plan (Editable)",
            value=st.session_state.treatment_plan,
            height=400,
            key="treatment_plan_editor"
        )
        
        st.session_state.treatment_plan = treatment_plan
        
        st.divider()
        
        clinician_notes = st.text_area(
            "Additional Clinician Notes",
            placeholder="Add any additional notes, modifications, or considerations...",
            height=150,
            key="clinician_notes"
        )
        
        st.divider()
        
        st.warning("⚠️ **Clinician Sign-off Required**")
        signed_off = st.checkbox(
            "I have reviewed the AI-generated findings and treatment plan, made necessary modifications, "
            "and take full clinical responsibility for this report.",
            key="signoff"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Plan", type="secondary", key="save_plan"):
                st.success("✅ Treatment plan saved!")
        
        with col2:
            if signed_off:
                if st.button("📄 Proceed to Export", type="primary", key="proceed_export"):
                    st.session_state.step = 5
                    st.session_state.clinician_notes = clinician_notes
                    st.session_state.signed_off = True
                    st.rerun()
            else:
                st.button("📄 Proceed to Export", type="primary", disabled=True, key="proceed_export_disabled")
                st.caption("⚠️ Sign-off required to proceed")

# TAB 5: Export
with tab5:
    st.header("Step 5: Export Final Report")
    
    if not st.session_state.get('signed_off', False):
        st.warning("⚠️ Clinician sign-off required before export!")
    else:
        st.success("✅ Report ready for export")
        
        # Generate report content
        report = f"""
{'='*70}
AI X-RAY DIAGNOSTIC REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER: Research prototype - Requires clinician review and signoff

{'='*70}
PATIENT CONTEXT
{'='*70}
Age: {st.session_state.patient_context.get('age', 'Not provided')}
Sex: {st.session_state.patient_context.get('sex', 'Not provided')}
Symptoms: {st.session_state.patient_context.get('symptoms', 'Not provided')}
Smoking Status: {st.session_state.patient_context.get('smoking_status', 'Not provided')}

VITAL SIGNS:
Temperature: {st.session_state.patient_context.get('vitals', {}).get('temp', 'N/A')}
Blood Pressure: {st.session_state.patient_context.get('vitals', {}).get('bp', 'N/A')}
Heart Rate: {st.session_state.patient_context.get('vitals', {}).get('hr', 'N/A')}
Respiratory Rate: {st.session_state.patient_context.get('vitals', {}).get('rr', 'N/A')}
SpO2: {st.session_state.patient_context.get('vitals', {}).get('spo2', 'N/A')}

{'='*70}
AI FINDINGS
{'='*70}
Model: {st.session_state.results['model_version']}
AUROC: {st.session_state.results['auroc']}
Analysis Timestamp: {st.session_state.results['timestamp']}

"""
        
        for i, finding in enumerate(st.session_state.results['findings'], 1):
            report += f"\n{i}. {finding['name']}\n"
            report += f"   Confidence: {finding['confidence']*100:.0f}%\n"
            report += f"   Severity: {finding['severity']}\n"
            report += f"   Description: {finding['description']}\n"
        
        report += f"\n{'='*70}\n"
        report += "DIFFERENTIAL DIAGNOSES\n"
        report += f"{'='*70}\n"
        for i, dx in enumerate(st.session_state.results['differentials'], 1):
            report += f"{i}. {dx}\n"
        
        report += f"\n{'='*70}\n"
        report += "TREATMENT PLAN (Clinician-Edited)\n"
        report += f"{'='*70}\n"
        report += st.session_state.treatment_plan + "\n"
        
        report += f"\n{'='*70}\n"
        report += "CLINICIAN NOTES\n"
        report += f"{'='*70}\n"
        report += st.session_state.get('clinician_notes', 'None') + "\n"
        
        report += f"\n{'='*70}\n"
        report += "AUDIT TRAIL\n"
        report += f"{'='*70}\n"
        report += f"Model Version: {st.session_state.results['model_version']}\n"
        report += f"Analysis Timestamp: {st.session_state.results['timestamp']}\n"
        report += f"Report Generated: {datetime.now().isoformat()}\n"
        report += f"Clinician Sign-off: YES\n"
        report += f"\nThis report was generated with AI assistance and has been\n"
        report += f"reviewed by a licensed clinician.\n"
        report += f"{'='*70}\n"
        
        # Display report
        st.text_area("Final Report", value=report, height=400, key="final_report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as text file
            st.download_button(
                label="📥 Download as TXT",
                data=report,
                file_name=f"xray_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="primary"
            )
        
        with col2:
            # Copy to clipboard
            if st.button("📋 Copy to Clipboard", type="secondary"):
                st.code(report, language=None)
                st.info("Report displayed above - use your browser's copy function")
        
        with col3:
            # Reset app
            if st.button("🔄 Start New Analysis", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>AI X-ray Diagnostic Reader v1.0</strong> | Research Prototype Only</p>
    <p style='font-size: 0.9em;'>Not for clinical use without physician oversight | All decisions require clinician approval</p>
    <p style='font-size: 0.8em; margin-top: 1rem;'>
        Model: DenseNet-121-CheXpert | AUROC: 0.87 | 
        <a href='#'>Documentation</a> | <a href='#'>Support</a>
    </p>
</div>
""", unsafe_allow_html=True)
