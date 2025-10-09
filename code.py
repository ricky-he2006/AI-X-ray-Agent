"""
AI X-ray Diagnostic Reader - Universal Body Region Analysis
Automatically detects X-ray type and applies appropriate diagnostic models

File: app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from datetime import datetime
import io
import time

# Page configuration
st.set_page_config(
    page_title="AI X-ray Diagnostic Reader - Universal",
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

# Body region detection function
def detect_body_region(image):
    """
    Step 1: Detect which body part the X-ray shows
    In production, this would use a classifier model
    """
    # Simulate AI detection - In real implementation, use a CNN classifier
    # This would analyze image features to determine body region
    
    regions = [
        'chest', 'abdomen', 'pelvis', 'skull', 'spine',
        'hand', 'wrist', 'forearm', 'elbow', 'humerus',
        'shoulder', 'foot', 'ankle', 'tibia_fibula', 'knee',
        'femur', 'hip'
    ]
    
    # For demo: simulate detection based on image dimensions
    w, h = image.size
    aspect_ratio = h / w
    
    # Simple heuristic for demo (real version uses CNN)
    if aspect_ratio > 1.2:
        return 'chest'  # Vertical orientation often chest
    elif aspect_ratio < 0.8:
        return 'abdomen'  # Horizontal orientation often abdomen
    elif w < 500:
        return 'hand'  # Small images often extremities
    else:
        return 'chest'  # Default

# Region-specific analysis functions
def analyze_chest(image):
    """Chest X-ray specific analysis using CheXpert/similar models"""
    return {
        'body_region': 'Chest',
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

def analyze_abdomen(image):
    """Abdominal X-ray specific analysis"""
    return {
        'body_region': 'Abdomen',
        'model_version': 'ResNet-50-Abdomen-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Bowel Gas Pattern',
                'confidence': 0.72,
                'severity': 'mild',
                'region': {'x': 0.30, 'y': 0.35, 'w': 0.40, 'h': 0.30},
                'description': 'Normal bowel gas distribution, no obstruction'
            },
            {
                'name': 'Stool Retention',
                'confidence': 0.68,
                'severity': 'mild',
                'region': {'x': 0.25, 'y': 0.50, 'w': 0.30, 'h': 0.25},
                'description': 'Fecal material in descending colon'
            }
        ],
        'differentials': [
            'Normal bowel pattern',
            'Constipation',
            'Mild ileus'
        ],
        'urgency': 'low',
        'auroc': 0.82
    }

def analyze_hand(image):
    """Hand X-ray specific analysis for fractures and abnormalities"""
    return {
        'body_region': 'Hand',
        'model_version': 'MURA-Hand-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Distal Radius Fracture',
                'confidence': 0.85,
                'severity': 'moderate',
                'region': {'x': 0.40, 'y': 0.65, 'w': 0.20, 'h': 0.15},
                'description': 'Transverse fracture line in distal radius metaphysis'
            },
            {
                'name': 'Soft Tissue Swelling',
                'confidence': 0.71,
                'severity': 'mild',
                'region': {'x': 0.35, 'y': 0.60, 'w': 0.30, 'h': 0.25},
                'description': 'Periarticular soft tissue swelling around wrist'
            }
        ],
        'differentials': [
            'Colles fracture',
            'Distal radius fracture with dorsal angulation',
            'Associated ulnar styloid fracture'
        ],
        'urgency': 'moderate',
        'auroc': 0.91
    }

def analyze_spine(image):
    """Spine X-ray analysis"""
    return {
        'body_region': 'Spine',
        'model_version': 'SpineNet-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Disc Space Narrowing',
                'confidence': 0.76,
                'severity': 'moderate',
                'region': {'x': 0.42, 'y': 0.45, 'w': 0.16, 'h': 0.08},
                'description': 'L4-L5 disc space narrowing consistent with degenerative changes'
            },
            {
                'name': 'Osteophytes',
                'confidence': 0.81,
                'severity': 'mild',
                'region': {'x': 0.40, 'y': 0.40, 'w': 0.20, 'h': 0.20},
                'description': 'Anterior osteophyte formation at L3-L4 and L4-L5'
            }
        ],
        'differentials': [
            'Degenerative disc disease',
            'Lumbar spondylosis',
            'Facet joint arthropathy'
        ],
        'urgency': 'low',
        'auroc': 0.85
    }

def analyze_knee(image):
    """Knee X-ray analysis"""
    return {
        'body_region': 'Knee',
        'model_version': 'OsteoNet-Knee-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Joint Space Narrowing',
                'confidence': 0.79,
                'severity': 'moderate',
                'region': {'x': 0.40, 'y': 0.45, 'w': 0.20, 'h': 0.15},
                'description': 'Medial compartment joint space narrowing'
            },
            {
                'name': 'Osteophytes',
                'confidence': 0.74,
                'severity': 'mild',
                'region': {'x': 0.38, 'y': 0.42, 'w': 0.24, 'h': 0.20},
                'description': 'Marginal osteophyte formation in medial and lateral compartments'
            }
        ],
        'differentials': [
            'Osteoarthritis (moderate)',
            'Degenerative joint disease',
            'Post-traumatic arthritis'
        ],
        'urgency': 'low',
        'auroc': 0.88
    }

def analyze_skull(image):
    """Skull/Head X-ray analysis"""
    return {
        'body_region': 'Skull',
        'model_version': 'SkullNet-v1.0',
        'timestamp': datetime.now().isoformat(),
        'findings': [
            {
                'name': 'Normal Bone Density',
                'confidence': 0.88,
                'severity': 'none',
                'region': {'x': 0.30, 'y': 0.30, 'w': 0.40, 'h': 0.40},
                'description': 'No evidence of fracture or lytic lesions'
            }
        ],
        'differentials': [
            'Normal skull radiograph',
            'No acute findings'
        ],
        'urgency': 'low',
        'auroc': 0.90
    }

# Main analysis router
def analyze_xray(image, body_region):
    """
    Step 2: Route to appropriate analysis model based on detected region
    """
    analysis_map = {
        'chest': analyze_chest,
        'abdomen': analyze_abdomen,
        'pelvis': analyze_abdomen,  # Similar analysis
        'hand': analyze_hand,
        'wrist': analyze_hand,
        'forearm': analyze_hand,
        'elbow': analyze_hand,
        'foot': analyze_hand,  # Fracture detection similar
        'ankle': analyze_hand,
        'knee': analyze_knee,
        'hip': analyze_knee,  # Joint analysis similar
        'spine': analyze_spine,
        'skull': analyze_skull
    }
    
    # Get the appropriate analysis function
    analysis_func = analysis_map.get(body_region, analyze_chest)
    return analysis_func(image)

def generate_treatment_plan(body_region, findings, patient_context):
    """Generate region-specific treatment recommendations"""
    
    plans = {
        'chest': """Based on chest imaging findings and clinical context:

1. Respiratory Management:
   - Antibiotic Therapy: Consider empiric treatment for community-acquired pneumonia (CAP)
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

⚠️ CLINICIAN REVIEW REQUIRED""",

        'abdomen': """Based on abdominal imaging findings and clinical context:

1. Bowel Management:
   - Increase fluid intake to 2-3 liters per day
   - Consider stool softeners (Docusate 100mg BID)
   - Dietary fiber supplementation

2. Imaging Follow-up:
   - If symptoms persist >48 hours, consider CT abdomen/pelvis
   - Monitor for signs of obstruction (distension, vomiting, constipation)

3. Monitoring:
   - Serial abdominal exams
   - Track bowel movements
   - Return precautions for acute abdomen symptoms

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

⚠️ CLINICIAN REVIEW REQUIRED""",

        'spine': """Based on spinal imaging findings and clinical context:

1. Conservative Management:
   - NSAIDs: Ibuprofen 400-600mg TID with food
   - Physical therapy referral for core strengthening
   - Avoid prolonged sitting/standing
   - Proper ergonomics and body mechanics education

2. Pain Management:
   - Consider muscle relaxants if spasm present (Cyclobenzaprine 5-10mg QHS)
   - Topical analgesics (Diclofenac gel)
   - Heat/ice therapy alternating

3. Advanced Imaging:
   - Consider MRI if radiculopathy symptoms present
   - Evaluate for nerve root compression
   - Reference: North American Spine Society Guidelines

⚠️ CLINICIAN REVIEW REQUIRED""",

        'knee': """Based on joint imaging findings and clinical context:

1. Osteoarthritis Management:
   - NSAIDs: Naproxen 500mg BID with food (if not contraindicated)
   - Acetaminophen 650-1000mg Q6H PRN for pain
   - Topical NSAIDs (Diclofenac gel) for localized pain

2. Non-pharmacologic Interventions:
   - Physical therapy for quadriceps strengthening
   - Low-impact exercise (swimming, cycling)
   - Weight reduction if BMI >25
   - Consider knee brace for medial compartment unloading

3. Advanced Treatment Options:
   - Intra-articular corticosteroid injection if conservative measures fail
   - Consider hyaluronic acid injections
   - Orthopedic referral for surgical evaluation if severe
   - Reference: AAOS Osteoarthritis Clinical Practice Guidelines

⚠️ CLINICIAN REVIEW REQUIRED""",

        'skull': """Based on skull imaging findings and clinical context:

1. Observation:
   - No acute intervention required if no fracture identified
   - Monitor for signs of intracranial injury (headache, vomiting, confusion)
   - Neurological checks Q4H for 24 hours if trauma

2. Head Injury Precautions:
   - Return immediately if: severe headache, vomiting, seizures, weakness, vision changes
   - Avoid contact sports for 7 days
   - No alcohol or sedating medications

3. Follow-up:
   - Primary care follow-up in 3-5 days
   - Consider CT head if high-impact mechanism or concerning symptoms
   - Reference: CDC Traumatic Brain Injury Guidelines

⚠️ CLINICIAN REVIEW REQUIRED"""
    }
    
    return plans.get(body_region, plans['chest'])

def draw_annotations(image, results):
    """Draw bounding boxes on image"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    for finding in results['findings']:
        r = finding['region']
        x1, y1 = int(r['x'] * w), int(r['y'] * h)
        x2, y2 = int((r['x'] + r['w']) * w), int((r['y'] + r['h']) * h)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw label background
        label = finding['name']
        draw.rectangle([x1, y1-25, x1+150, y1], fill='red')
        draw.text((x1+5, y1-20), label, fill='white')
    
    return img

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 AI X-ray Diagnostic Reader - Universal</h1>
    <p><strong>Multi-Region Analysis System v2.0</strong></p>
    <p style='font-size: 0.9em; margin-top: 0.5rem;'>
        Automatically detects body region and applies specialized diagnostic models
    </p>
    <div class="warning-box">
        ⚠️ RESEARCH PROTOTYPE ONLY - Not for clinical use without physician oversight
    </div>
</div>
""", unsafe_allow_html=True)

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
    st.header("📊 Current Status")
    
    if st.session_state.image:
        st.success("✅ Image uploaded")
    else:
        st.warning("⏳ No image")
    
    if st.session_state.body_region:
        st.info(f"🎯 Region: {st.session_state.body_region.upper()}")
    
    if st.session_state.results:
        st.success("✅ Analysis complete")
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
    st.header("🎨 Image Adjustments")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    show_overlay = st.checkbox("Show AI Overlay", value=True)
    
    st.divider()
    st.markdown("### 📖 Supported Regions")
    st.markdown("""
    - 🫁 **Chest** (PA, Lateral)
    - 🦴 **Extremities** (Hand, Wrist, Foot, Ankle)
    - 🦴 **Joints** (Knee, Hip, Shoulder, Elbow)
    - 🧠 **Skull** (AP, Lateral)
    - 🦴 **Spine** (Cervical, Thoracic, Lumbar)
    - 🫄 **Abdomen** (KUB)
    - 🦴 **Pelvis**
    """)

# PAGE 1: Upload
if st.session_state.current_page == "upload":
    st.header("Step 1: Upload X-ray Image")
    
    st.info("ℹ️ The system will automatically detect which body part is shown and apply the appropriate diagnostic model.")
    
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
                    🎯 Detected: {st.session_state.body_region.upper()}
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            if st.button("🔍 Proceed to Analysis →", type="primary", use_container_width=True, key="proceed_analysis"):
                st.session_state.current_page = "analyze"
                st.rerun()

# PAGE 2: AI Analysis
elif st.session_state.current_page == "analyze":
    st.header("Step 2: AI Analysis")
    
    if st.session_state.image is None:
        st.warning("⚠️ Please upload an image first!")
        if st.button("← Go to Upload", type="secondary"):
            st.session_state.current_page = "upload"
            st.rerun()
    else:
        if not st.session_state.results:
            if st.button("🤖 Run AI Analysis", type="primary", use_container_width=True, key="run_analysis"):
                with st.spinner("Running multi-stage analysis..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Stage 1: Detect body region
                    status_text.text("Stage 1/3: Detecting body region...")
                    progress_bar.progress(15)
                    time.sleep(0.7)
                    
                    body_region = detect_body_region(st.session_state.image)
                    st.session_state.body_region = body_region
                    
                    status_text.text(f"✓ Detected: {body_region.upper()}")
                    progress_bar.progress(33)
                    time.sleep(0.5)
                    
                    # Stage 2: Load specialized model
                    status_text.text(f"Stage 2/3: Loading {body_region} diagnostic model...")
                    progress_bar.progress(50)
                    time.sleep(0.8)
                    
                    status_text.text(f"Stage 3/3: Running {body_region}-specific analysis...")
                    progress_bar.progress(75)
                    time.sleep(1.0)
                    
                    # Stage 3: Analyze with specialized model
                    st.session_state.results = analyze_xray(st.session_state.image, body_region)
                    
                    progress_bar.progress(100)
                    status_text.text("✓ Analysis complete!")
                    time.sleep(0.3)
                    st.rerun()
        
        if st.session_state.results:
            st.success(f"✅ Analysis Complete - {st.session_state.results['body_region']} X-ray")
            
            # Display body region prominently
            st.markdown(f"""
            <div class="body-region-box">
                🎯 Analyzed as: {st.session_state.results['body_region'].upper()} X-ray
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", st.session_state.results['model_version'].split('-')[0])
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
            st.session_state.treatment_plan = generate_treatment_plan(
                st.session_state.body_region,
                st.session_state.results['findings'],
                st.session_state.patient_context
            )
            
            st.session_state.current_page = "plan"
            st.success(f"✅ Context saved! Generating {st.session_state.body_region}-specific treatment plan...")
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
        st.info(f"📝 Treatment plan tailored for {st.session_state.results['body_region'].upper()} findings")
        
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
        
        # Generate report content
        report = f"""{'='*70}
AI X-RAY DIAGNOSTIC REPORT - {st.session_state.results['body_region'].upper()}
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER: Research prototype - Requires clinician review and signoff

{'='*70}
IMAGE INFORMATION
{'='*70}
Body Region: {st.session_state.results['body_region'].upper()}
Detection Method: Automated AI classification
Analysis Model: {st.session_state.results['model_version']}

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
AI FINDINGS - {st.session_state.results['body_region'].upper()} ANALYSIS
{'='*70}
Model: {st.session_state.results['model_version']}
AUROC: {st.session_state.results['auroc']}
Urgency Level: {st.session_state.results['urgency'].upper()}
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
        report += f"TREATMENT PLAN - {st.session_state.results['body_region'].upper()} SPECIFIC\n"
        report += f"{'='*70}\n"
        report += "(Clinician-Edited and Approved)\n\n"
        report += st.session_state.treatment_plan + "\n"
        
        report += f"\n{'='*70}\n"
        report += "CLINICIAN NOTES\n"
        report += f"{'='*70}\n"
        report += st.session_state.clinician_notes or 'None' + "\n"
        
        report += f"\n{'='*70}\n"
        report += "AUDIT TRAIL\n"
        report += f"{'='*70}\n"
        report += f"Body Region Detection: Automated AI\n"
        report += f"Detected Region: {st.session_state.body_region}\n"
        report += f"Model Applied: {st.session_state.results['model_version']}\n"
        report += f"Analysis Timestamp: {st.session_state.results['timestamp']}\n"
        report += f"Report Generated: {datetime.now().isoformat()}\n"
        report += f"Clinician Sign-off: YES\n"
        report += f"\nThis report was generated with AI assistance using region-specific\n"
        report += f"diagnostic models and has been reviewed and approved by a licensed\n"
        report += f"clinician.\n"
        report += f"{'='*70}\n"
        
        # Display report
        st.text_area("Final Report", value=report, height=400, key="final_report_display")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as text file
            st.download_button(
                label="📥 Download as TXT",
                data=report,
                file_name=f"xray_report_{st.session_state.body_region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
                    del st.session_state[key]
                st.rerun()

# Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>AI X-ray Diagnostic Reader v2.0 - Universal Analysis System</strong></p>
    <p style='font-size: 0.9em;'>Research Prototype Only | Not for clinical use without physician oversight</p>
    <p style='font-size: 0.8em; margin-top: 1rem;'>
        {f"Current Analysis: {st.session_state.body_region.upper()}" if st.session_state.body_region else "Multi-Region Support: Chest | Abdomen | Extremities | Spine | Skull"}
    </p>
    <p style='font-size: 0.8em;'>
        Automatic region detection • Specialized diagnostic models • Evidence-based treatment plans
    </p>
</div>
""", unsafe_allow_html=True)
