import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
from login import main as login_main, init_db

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize the database
init_db()

# Function to detect facial features
def detect_facial_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Extract relevant facial features (e.g., eye shape, lip shape, face shape)
        # This is a simplified example; you may want to add more detailed feature extraction
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[4]
        left_lip = landmarks[61]
        right_lip = landmarks[291]
        
        features = {
            "eye_distance": np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2),
            "nose_to_lip_distance": np.sqrt((nose_tip.x - left_lip.x)**2 + (nose_tip.y - left_lip.y)**2),
            "lip_width": np.sqrt((left_lip.x - right_lip.x)**2 + (left_lip.y - right_lip.y)**2),
        }
        return features
    return None

# Function to get makeup and accessory recommendations
def get_recommendations(features):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Based on the following facial features, recommend makeup and beauty accessories:
    - Eye distance: {features['eye_distance']:.4f}
    - Nose to lip distance: {features['nose_to_lip_distance']:.4f}
    - Lip width: {features['lip_width']:.4f}
    
    Please provide specific recommendations for:
    1. Eye makeup
    2. Lip color and style
    3. Face contouring
    4. Accessories (e.g., earrings, necklaces)
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Facial Beauty Advisor")

# Check if user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Show login page if not logged in
if not st.session_state.logged_in:
    login_main()
else:
    # Initialize session state for image if not already done
    if 'image' not in st.session_state:
        st.session_state.image = None

    # Camera input
    camera_input = st.camera_input("Take a picture")

    if st.button("Save Image"):
        if st.session_state.image is not None:
            # Save logic here (e.g., to a file or database)
            st.success("Image saved successfully!")

    # Process the captured image
    if camera_input:
        st.session_state.image = camera_input.getvalue()
        
    if st.session_state.image:
        # Display the captured image
        st.image(st.session_state.image, caption="Captured Image", use_column_width=True)
        
        # Process the image and get recommendations
        image = cv2.imdecode(np.frombuffer(st.session_state.image, np.uint8), cv2.IMREAD_COLOR)
        features = detect_facial_features(image)
        
        if features:
            recommendations = get_recommendations(features)
            st.subheader("Makeup and Accessory Recommendations")
            st.write(recommendations)
        else:
            st.error("No face detected in the image. Please try again with a clear facial image.")
    else:
        st.info("Please take a photo to get makeup and accessory recommendations.")

    # Add a logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# Instructions
st.markdown("""
## How to use:
1. Click on "Take a picture" to capture your photo.
2. The app will analyze your facial features and provide personalized recommendations.
3. Use the "Save Image" button to save your favorite captured image.
""")
