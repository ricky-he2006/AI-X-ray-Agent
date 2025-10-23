# app.py
import streamlit as st
from PIL import Image
from ai_models import UniversalXrayAnalyzer

st.set_page_config(page_title="AI X-Ray Analyzer", layout="centered")

st.title("🩻 Universal AI X-Ray Analyzer")
st.markdown("Upload an X-ray image and the AI will identify the region and predict possible findings.")

uploaded_file = st.file_uploader("Upload an X-ray image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    st.write("Running AI model... please wait ⏳")

    try:
        analyzer = UniversalXrayAnalyzer()
        result = analyzer.analyze(image)

        st.subheader("🧠 Analysis Results")
        st.write(f"**Detected Region:** {result['body_region']} ({result['region_confidence']:.2f})")

        st.write("**Findings:**")
        for finding in result["findings"]:
            st.write(f"- {finding['condition']} ({finding['confidence']:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with PyTorch • Streamlit • OpenAI GPT-5")
