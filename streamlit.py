import streamlit as st
import requests

# FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8090/analyze_image/"  # Adjust the URL if needed

# Page configuration
st.set_page_config(
    page_title="Car Damage Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a sidebar for navigation and instructions
st.sidebar.title("Navigation")
st.sidebar.markdown("""
Use this application to:
- Upload a car image.
- Get an automated damage analysis report.
""")
st.sidebar.info("Ensure the image is in `.jpg` or `.png` format for best results.")

# Main page
st.title("🚗 Car Damage Analysis and Documentation")
st.write("Upload a car image to analyze damages and generate a structured report.")

# File upload section
st.header("🔍 Upload and Analyze")
uploaded_image = st.file_uploader(
    "Upload a car image (.jpg or .png):",
    type=("jpg", "png"),
    help="Supported formats: JPEG, PNG"
)

if uploaded_image:
    # Display the uploaded image in a centered column
    st.markdown("### Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image when the button is clicked
    analyze_button = st.button("🔍 Analyze Image")
    if analyze_button:
        with st.spinner("Analyzing image... Please wait."):
            # Prepare the file for sending to FastAPI
            files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}

            # Send the image to the FastAPI endpoint
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                
                # Extract and display only the final report
                if "analysis" in result:
                    final_report = result["analysis"]
                    
                    # Display report in an expandable section
                    st.success("Analysis completed successfully!")
                    st.markdown("### 📋 Damage Analysis Report")
                    with st.expander("Click to view the detailed report"):
                        st.write(final_report)
                else:
                    st.error("The response does not contain the expected 'analysis' field.")
            else:
                st.error(f"Error analyzing the image: {response.status_code} - {response.text}")

# Footer section
st.markdown("---")
st.markdown("""
**About this App:**
This tool leverages advanced computer vision and language models to analyze car damages and generate human-readable reports.
""")
