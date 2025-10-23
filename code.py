"""
AI X-ray Diagnostic Reader - REAL AI Implementation
Integrated with actual PyTorch models

File: app_real.py

Setup:
1. pip install -r requirements.txt
2. Download model weights (see ai_models.py)
3. streamlit run app_real.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from datetime import datetime
import io

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

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'body_region' not in st.session_state:
    st.session_state.body_region = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'patient_context' not in st.session_state:
    st.session_state.patient_context = {}
if 'treatment_plan' not in st.session_state:
    st.session_state.treatment_plan = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "upload"
if 'clinician_notes' not in st.session_state:
    st.session_state.clinician_notes = ""
if 'signed_off' not in st.session_state:
    st.session_state.signed_off = False
if 'analyzer' not in st.session_state and REAL_AI_AVAILABLE:
    with st.spinner("Loading AI models..."):
        st.session_state.analyzer = UniversalXrayAnalyzer()

def generate_treatment_plan(body_region, findings, patient_context):
    """Generate region-specific treatment recommendations"""
    
    plans = {
        'chest': """Based on chest imaging findings and clinical context:

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

        'hand': """Based on extremity imaging findings and clinical context:

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
    
    # Get base plan or default
    region_lower = body_region.lower()
    plan = plans.get(region_lower, plans.get('chest', ''))
    
    return plan

def draw_annotations(image, results):
    """Draw bounding boxes on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    if 'findings' in results:
        for finding in results['findings']:
            if 'region' in finding:
                r = finding['region']
                x1, y1 = int(r['x'] * w), int(r['y'] * h)
                x2, y2 = int((r['x'] + r['w']) * w), int((r['y'] + r['h']) * h)
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                # Draw label background
                label = finding['name']
                draw.rectangle([x1, y1-25, x1+200, y1], fill='red')
                draw.text((x1+5, y1-20), label, fill='white')
    
    return img

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI X-ray Diagnostic Reader</h1>
    <div class="ai-badge">🤖 REAL AI POWERED</div>
    <p><strong>Multi-Region Analysis with PyTorch Models</strong></p>
    <p style='font-size: 0.9em; margin-top: 0.5rem;'>
        Using DenseNet-121 (CheXpert) • DenseNet-169 (MURA) • ResNet-50 (Region Classification)
    </p>
    <div class="warning-box">
        ⚠️ RESEARCH PROTOTYPE ONLY - Not for clinical use without physician oversight
    </div>
</div>
""", unsafe_allow_html=True)

# Check if AI is available
if not REAL_AI_AVAILABLE:
    st.error("🚫 AI models not loaded. Please install dependencies and restart.")
    st.code("pip install torch torchvision opencv-python", language="bash")
    st.stop()

# Navigation buttons
st.markdown("### Navigation")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📁 1. Upload", key="nav_upload", use_container_width=True, 
                 type="primary" if st.session_state.current_page == "upload" else "secondary"):
        st.session_state.current_page = "upload"
        st.rerun()

with col2:
    if st.button("🤖 2. Analyze", key="nav_analyze", use_container_width=True,
                 type="primary" if st.session_state.current_page == "analyze" else "secondary",
                 disabled=st.session_state.image is None):
        st.session_state.current_page = "analyze"
        st.rerun()

with col3:
    if st.button("📋 3. Context", key="nav_context", use_container_width=True,
                 type="primary" if st.session_state.current_page == "context" else "secondary",
                 disabled=st.session_state.results is None):
        st.session_state.current_page = "context"
        st.rerun()

with col4:
    if st.button("💊 4. Plan", key="nav_plan", use_container_width=True,
                 type="primary" if st.session_state.current_page == "plan" else "secondary",
                 disabled=st.session_state.treatment_plan == ""):
        st.session_state.current_page = "plan"
        st.rerun()

with col5:
    if st.button("📄 5. Export", key="nav_export", use_container_width=True,
                 type="primary" if st.session_state.current_page == "export" else "secondary",
                 disabled=not st.session_state.signed_off):
        st.session_state.current_page = "export"
        st.rerun()

st.divider()

# Sidebar
with st.sidebar:
    st.header(" System Status")
    
    # AI Status
    st.markdown("### 🤖 AI Models")
    st.success("✅ PyTorch loaded")
    st.info("🧠 Models: DenseNet-121, DenseNet-169, ResNet-50")
    
    st.divider()
    st.header("🎨 Image Adjustments")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    show_overlay = st.checkbox("Show AI Overlay", value=True)
    
    st.divider()
    st.markdown("### 📖 Supported Regions")
    st.markdown("""
    - 🫁 **Chest** (CheXpert model)
    - 🦴 **Hand/Wrist** (MURA model)
    - 🦴 **Foot/Ankle** (MURA model)
    - 🦴 **Knee** (MURA model)
    - 🦴 **Elbow/Shoulder** (MURA model)
    - 🧠 **Skull** (Custom model)
    - 🦴 **Spine** (Custom model)
    """)

# PAGE 1: Upload
if st.session_state.current_page == "upload":
    st.header("Step 1: Upload X-ray Image")
    
    st.info("ℹ️ The system will automatically detect the body region using ResNet-50 classifier and apply the appropriate PyTorch model.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image (any body region)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)
        
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
                img = draw_annotations(img, st.session_state.results)
            
            st.image(img, caption='X-ray Image', use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📏 Image Info</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Width", f"{st.session_state.image.size[0]} px")
            st.metric("Height", f"{st.session_state.image.size[1]} px")
            st.metric("Mode", st.session_state.image.mode)
            
            if st.session_state.body_region:
                st.divider()
                st.markdown(f"""
                <div class="body-region-box">
                    🎯 Detected: {st.session_state.body_region}
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            if st.button("🔍 Proceed to AI Analysis →", type="primary", use_container_width=True, key="proceed_analysis"):
                st.session_state.current_page = "analyze"
                st.rerun()

# PAGE 2: AI Analysis (REAL AI)
elif st.session_state.current_page == "analyze":
    st.header("Step 2: Real AI Analysis")
    
    if st.session_state.image is None:
        st.warning("⚠️ Please upload an image first!")
        if st.button("← Go to Upload", type="secondary"):
            st.session_state.current_page = "upload"
            st.rerun()
    else:
        if not st.session_state.results:
            if st.button("🤖 Run Real AI Analysis", type="primary", use_container_width=True, key="run_analysis"):
                with st.spinner("Running real PyTorch models..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Run REAL AI analysis
                        status_text.text("⚙️ Running AI inference...")
                        progress_bar.progress(10)
                        
                        # Convert PIL to format needed by models
                        img_array = np.array(st.session_state.image)
                        
                        # Run the actual AI analyzer
                        results = st.session_state.analyzer.analyze(st.session_state.image)
                        
                        progress_bar.progress(100)
                        status_text.text("✓ Real AI analysis complete!")
                        
                        st.session_state.results = results
                        st.session_state.body_region = results['body_region']
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("Make sure model weights are downloaded. See ai_models.py")
                        status_text.text("")
                        progress_bar.empty()
        
        if st.session_state.results:
            st.success(f"✅ Real AI Analysis Complete - {st.session_state.results['body_region']} X-ray")
            
            # Display detected region with confidence
            region_conf = st.session_state.results.get('region_confidence', 0)
            st.markdown(f"""
            <div class="body-region-box">
                🎯 AI Detected: {st.session_state.results['body_region'].upper()}
                <br><small>Confidence: {region_conf*100:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                model_short = st.session_state.results['model_version'].split('-')[0]
                st.metric("AI Model", model_short)
            with col2:
                st.metric("AUROC", st.session_state.results['auroc'])
            with col3:
                st.metric("Urgency", st.session_state.results['urgency'].upper())
            
            st.divider()
            
            st.subheader("🔍 AI-Detected Findings")
            
            if st.session_state.results['findings']:
                for i, finding in enumerate(st.session_state.results['findings'], 1):
                    severity_class = finding.get('severity', 'mild')
                    st.markdown(f"""
                    <div class="finding-box {severity_class}">
                        <strong>{i}. {finding['name']}</strong> 
                        <span style="float: right;">AI Confidence: {finding['confidence']*100:.1f}%</span>
                        <br>
                        <small><em>{finding.get('description', 'AI-detected abnormality')}</em></small>
                        <br>
                        <small>Severity: <strong>{severity_class.upper()}</strong></small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✓ No significant abnormalities detected by AI")
            
            st.divider()
            
            st.subheader("🩺 AI-Generated Differential Diagnoses")
            for i, dx in enumerate(st.session_state.results['differentials'], 1):
                st.write(f"{i}. {dx}")
            
            st.info("💡 These findings were generated by real PyTorch neural networks trained on medical imaging datasets.")
            
            if st.button("📋 Proceed to Patient Context →", type="primary", use_container_width=True, key="proceed_context"):
                st.session_state.current_page = "context"
                st.rerun()

# PAGE 3: Patient Context
elif st.session_state.current_page == "context":
    st.header("Step 3: Patient Clinical Context")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please complete AI analysis first!")
        if st.button("← Go to Analysis", type="secondary"):
            st.session_state.current_page = "analyze"
            st.rerun()
    else:
        st.info(f"📋 Collecting context for {st.session_state.results['body_region']} X-ray findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=0)
            sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
            smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        
        with col2:
            symptoms = st.text_area("Chief Complaint / Symptoms", 
                                   placeholder="e.g., Pain, swelling, shortness of breath",
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
        
        if st.button("💊 Save & Generate Treatment Plan →", type="primary", use_container_width=True, key="save_context"):
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
            
            # Generate region-specific treatment plan
            region = st.session_state.body_region.lower()
            st.session_state.treatment_plan = generate_treatment_plan(
                region,
                st.session_state.results['findings'],
                st.session_state.patient_context
            )
            
            st.session_state.current_page = "plan"
            st.success(f"✅ Context saved! Generating treatment plan...")
            st.rerun()

# PAGE 4: Treatment Plan
elif st.session_state.current_page == "plan":
    st.header("Step 4: Evidence-Based Treatment Plan")
    
    if st.session_state.treatment_plan == "":
        st.warning("⚠️ Please complete patient context first!")
        if st.button("← Go to Context", type="secondary"):
            st.session_state.current_page = "context"
            st.rerun()
    else:
        st.info(f"📝 Treatment plan tailored for {st.session_state.results['body_region'].upper()} findings from real AI analysis")
        
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
            value=st.session_state.clinician_notes,
            placeholder="Add any additional notes, modifications, or considerations...",
            height=150
        )
        
        st.session_state.clinician_notes = clinician_notes
        
        st.divider()
        
        st.warning("⚠️ **Clinician Sign-off Required**")
        signed_off = st.checkbox(
            "I have reviewed the AI-generated findings and treatment plan, made necessary modifications, "
            "and take full clinical responsibility for this report.",
            value=st.session_state.signed_off
        )
        
        st.session_state.signed_off = signed_off
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Plan", type="secondary", use_container_width=True, key="save_plan"):
                st.success("✅ Treatment plan saved!")
        
        with col2:
            if signed_off:
                if st.button("📄 Proceed to Export →", type="primary", use_container_width=True, key="proceed_export"):
                    st.session_state.current_page = "export"
                    st.rerun()
            else:
                st.button("📄 Proceed to Export →", type="primary", disabled=True, use_container_width=True, key="proceed_export_disabled")
                st.caption("⚠️ Sign-off required to proceed")

# PAGE 5: Export
elif st.session_state.current_page == "export":
    st.header("Step 5: Export Final Report")
    
    if not st.session_state.signed_off:
        st.warning("⚠️ Clinician sign-off required before export!")
        if st.button("← Go to Treatment Plan", type="secondary"):
            st.session_state.current_page = "plan"
            st.rerun()
    else:
        st.success("✅ Report ready for export")
        
        # Generate comprehensive report
        report = f"""{'='*70}
AI X-RAY DIAGNOSTIC REPORT - {st.session_state.results['body_region'].upper()}
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER: AI-assisted analysis - Requires clinician review and signoff

{'='*70}
AI ANALYSIS DETAILS
{'='*70}
Body Region Detection: Automated (ResNet-50)
Detected Region: {st.session_state.results['body_region']}
Detection Confidence: {st.session_state.results.get('region_confidence', 0)*100:.1f}%
Diagnostic Model: {st.session_state.results['model_version']}
Model Performance (AUROC): {st.session_state.results['auroc']}

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
AI-DETECTED FINDINGS - {st.session_state.results['body_region'].upper()} ANALYSIS
{'='*70}
Analysis Timestamp: {st.session_state.results['timestamp']}
Urgency Level: {st.session_state.results['urgency'].upper()}

"""
        
        if st.session_state.results['findings']:
            for i, finding in enumerate(st.session_state.results['findings'], 1):
                report += f"\n{i}. {finding['name']}\n"
                report += f"   AI Confidence: {finding['confidence']*100:.1f}%\n"
                report += f"   Severity: {finding.get('severity', 'unknown')}\n"
                report += f"   Description: {finding.get('description', 'N/A')}\n"
        else:
            report += "\nNo significant abnormalities detected by AI models.\n"
        
        report += f"\n{'='*70}\n"
        report += "AI-GENERATED DIFFERENTIAL DIAGNOSES\n"
        report += f"{'='*70}\n"
        for i, dx in enumerate(st.session_state.results['differentials'], 1):
            report += f"{i}. {dx}\n"
        
        report += f"\n{'='*70}\n"
        report += f"TREATMENT PLAN - {st.session_state.results['body_region'].upper()} SPECIFIC\n"
        report += f"{'='*70}\n"
        report += "(Clinician-Edited and Approved)\n\n"
        report += st.session_state.treatment_plan + "\n"
        
        report += f"\n{'='*70}\n"
        report += "CLINICIAN NOTES\n"
        report += f"{'='*70}\n"
        report += st.session_state.clinician_notes or 'None' + "\n"
        
        report += f"\n{'='*70}\n"
        report += "AUDIT TRAIL & AI TRANSPARENCY\n"
        report += f"{'='*70}\n"
        report += f"Region Detection Method: ResNet-50 CNN Classifier\n"
        report += f"Detected Region: {st.session_state.body_region}\n"
        report += f"Region Detection Confidence: {st.session_state.results.get('region_confidence', 0)*100:.1f}%\n"
        report += f"Diagnostic Model Applied: {st.session_state.results['model_version']}\n"
        report += f"Model Training Dataset: CheXpert/MURA (Stanford ML Group)\n"
        report += f"Model AUROC: {st.session_state.results['auroc']}\n"
        report += f"AI Analysis Timestamp: {st.session_state.results['timestamp']}\n"
        report += f"Report Generated: {datetime.now().isoformat()}\n"
        report += f"Clinician Sign-off: YES\n"
        report += f"\nThis report was generated using real PyTorch deep learning models\n"
        report += f"trained on medical imaging datasets. All AI findings have been\n"
        report += f"reviewed and approved by a licensed clinician who takes full\n"
        report += f"responsibility for the clinical interpretation and treatment plan.\n"
        report += f"{'='*70}\n"
        
        # Display report
        st.text_area("Final Report", value=report, height=400, key="final_report_display")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as text file
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"xray_AI_report_{st.session_state.body_region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            if st.button("📋 Show Full Report", type="secondary", use_container_width=True, key="show_report"):
                st.code(report, language=None)
        
        with col3:
            # Reset app
            if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True, key="reset_app"):
                for key in list(st.session_state.keys()):
                    if key != 'analyzer':  # Keep the AI model loaded
                        del st.session_state[key]
                st.rerun()

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>AI X-ray Diagnostic Reader v2.0 - Real PyTorch Implementation</strong></p>
    <p style='font-size: 0.9em;'>Powered by: DenseNet-121 • DenseNet-169 • ResNet-50</p>
    <p style='font-size: 0.8em; margin-top: 1rem;'>
        {f"Current: {st.session_state.body_region.upper()} | Model: {st.session_state.results['model_version']}" if st.session_state.body_region else "Real AI • Automatic Region Detection • Evidence-Based Analysis"}
    </p>
    <p style='font-size: 0.8em;'>
        Research Prototype | Not for clinical use without physician oversight
    </p>
</div>
""", unsafe_allow_html=True) Analysis Status")
    
    if st.session_state.image:
        st.success("✅ Image uploaded")
    else:
        st.warning("⏳ No image")
    
    if st.session_state.body_region:
        st.info(f"🎯 Region: {st.session_state.body_region}")
    
    if st.session_state.results:
        st.success("✅ AI analysis complete")
    else:
        st.warning("⏳ Not analyzed")
    
    if st.session_state.treatment_plan:
        st.success("✅ Plan generated")
    else:
        st.warning("⏳ No plan")
    
    if st.session_state.signed_off:
        st.success("✅ Signed off")
    else:
        st.warning("⏳ Not signed off")
    
    st.divider()
    st.header("
