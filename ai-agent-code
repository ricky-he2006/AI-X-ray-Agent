import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# Configuration
EDGE_FUNCTION_URL = "https://iasahsykxifpqejjsawk.supabase.co/functions/v1/analyze-xray"

# Page config
st.set_page_config(
    page_title="AI X-Ray Analyzer",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .finding-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .critical {
        border-left-color: #dc3545;
    }
    .moderate {
        border-left-color: #fd7e14;
    }
    .mild {
        border-left-color: #20c997;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🏥 AI X-Ray Analyzer")
st.markdown("Advanced medical imaging analysis powered by AI")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = None
if 'xray_image' not in st.session_state:
    st.session_state.xray_image = None

# Step 1: Patient Information
st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.header("Step 1: Patient Information")

with st.form("patient_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Patient Name *", placeholder="John Doe")
        age = st.number_input("Age *", min_value=0, max_value=120, value=35)
    
    with col2:
        gender = st.selectbox("Gender *", ["", "Male", "Female", "Other"])
    
    symptoms = st.text_area("Current Symptoms", 
                           placeholder="Describe current symptoms (e.g., chest pain, difficulty breathing...)")
    
    medical_history = st.text_area("Medical History",
                                   placeholder="Any relevant medical history...")
    
    medications = st.text_area("Current Medications",
                              placeholder="List current medications...")
    
    col1, col2 = st.columns(2)
    with col1:
        submit_with_info = st.form_submit_button("Continue with Patient Info", use_container_width=True)
    with col2:
        skip_info = st.form_submit_button("Skip (Analysis Only)", use_container_width=True)
    
    if submit_with_info and name and age and gender:
        st.session_state.patient_info = {
            "name": name,
            "age": str(age),
            "gender": gender.lower(),
            "symptoms": symptoms,
            "medicalHistory": medical_history,
            "currentMedications": medications
        }
        st.success(f"Patient info saved for {name}")
    
    if skip_info:
        st.session_state.patient_info = None
        st.info("Continuing without patient information")

st.markdown('</div>', unsafe_allow_html=True)

# Step 2: X-Ray Upload
st.markdown('<div class="result-card">', unsafe_allow_html=True)
st.header("Step 2: Upload X-Ray Image")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.session_state.xray_image = image
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with col2:
        st.info("📊 Image loaded successfully!")
        st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"**Format:** {image.format}")
        
        if st.button("🔍 Analyze X-Ray", use_container_width=True, type="primary"):
            with st.spinner("AI is analyzing the X-ray... This may take 30-60 seconds..."):
                try:
                    # Convert image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    img_data = f"data:image/png;base64,{img_str}"
                    
                    # Prepare payload
                    payload = {
                        "imageData": img_data,
                        "patientInfo": st.session_state.patient_info
                    }
                    
                    # Call edge function
                    response = requests.post(
                        EDGE_FUNCTION_URL,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        st.session_state.analysis_result = response.json()
                        st.success("✅ Analysis complete!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Display Results
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    
    # Header Card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.header("📋 Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Body Region", result['body_region'])
    with col2:
        st.metric("Confidence", f"{result['region_confidence']*100:.1f}%")
    with col3:
        urgency_emoji = {"critical": "🚨", "moderate": "⚠️", "low": "✅"}
        st.metric("Urgency", f"{urgency_emoji.get(result['urgency'], '❓')} {result['urgency'].upper()}")
    with col4:
        st.metric("Model Accuracy", f"{result['auroc']*100:.1f}%")
    
    st.markdown(f"**Model:** {result['model_version']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Findings
    if result.get('findings'):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader(f"🔍 Clinical Findings ({len(result['findings'])})")
        
        for finding in result['findings']:
            severity_class = finding['severity']
            severity_color = {
                'critical': '🔴',
                'moderate': '🟠',
                'mild': '🟢',
                'none': '⚪'
            }
            
            st.markdown(f"""
            <div class="finding-card {severity_class}">
                <h4>{severity_color.get(finding['severity'], '❓')} {finding['name']}</h4>
                <p><strong>Severity:</strong> {finding['severity'].upper()}</p>
                <p><strong>Confidence:</strong> {finding['confidence']*100:.1f}%</p>
                <p>{finding['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Differential Diagnoses
    if result.get('differentials'):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📊 Differential Diagnoses")
        for i, diff in enumerate(result['differentials'], 1):
            st.markdown(f"{i}. {diff}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Treatment Plan
    if result.get('treatment_plan'):
        tp = result['treatment_plan']
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("💊 Recommended Treatment Plan")
        
        if st.session_state.patient_info:
            st.info(f"Personalized for: {st.session_state.patient_info['name']}, {st.session_state.patient_info['age']} years old")
        
        # Immediate Actions
        if tp.get('immediate_actions'):
            st.markdown("#### 🚨 Immediate Actions Required")
            for action in tp['immediate_actions']:
                st.markdown(f"- {action}")
        
        # Recommended Tests
        if tp.get('recommended_tests'):
            st.markdown("#### 🔬 Additional Tests Recommended")
            for test in tp['recommended_tests']:
                st.markdown(f"- {test}")
        
        # Medications
        if tp.get('medication_suggestions'):
            st.markdown("#### 💊 Medication Considerations")
            for med in tp['medication_suggestions']:
                st.markdown(f"- {med}")
        
        # Lifestyle
        if tp.get('lifestyle_recommendations'):
            st.markdown("#### ❤️ Lifestyle Recommendations")
            for rec in tp['lifestyle_recommendations']:
                st.markdown(f"- {rec}")
        
        # Follow-up
        if tp.get('follow_up'):
            st.markdown("#### 📅 Follow-up Schedule")
            st.markdown(tp['follow_up'])
        
        # Specialist
        if tp.get('specialist_referral'):
            st.markdown("#### 👨‍⚕️ Specialist Consultation")
            st.markdown(tp['specialist_referral'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Start Over Button
    if st.button("🔄 Analyze Another X-Ray", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.patient_info = None
        st.session_state.xray_image = None
        st.rerun()
    
    # Disclaimer
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.warning("""
    ⚠️ **Medical Disclaimer:** This AI analysis and treatment recommendations are for 
    educational and research purposes only. They should not be used as a substitute for 
    professional medical advice, diagnosis, or treatment. Always consult with qualified 
    healthcare professionals before making any medical decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Powered by Lovable AI • For research and educational purposes only")
