import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-Ray",
    page_icon="🩺",
    layout="wide"
)

# Check TensorFlow version
st.write(f"TensorFlow Version: {tf.__version__}")

# Title and description
st.title("Pneumonia Detection from Chest X-Ray")
st.markdown("""
This application analyzes chest X-ray images to detect pneumonia.
Upload a chest X-ray image to get instant analysis results.
""")

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model input"""
    try:
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize and normalize
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_prediction(model, image):
    """Get model prediction"""
    try:
        prediction = model.predict(image)
        return float(prediction[0][0])
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_column_width=True)
    
    with col2:
        try:
            # Load and preprocess the image
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            if processed_image is None:
                st.stop()
            
            # Load model
            model_path = r'model/chest_xray_model2.h5'
            st.subheader("Analysis Results")

            if Path(model_path).exists():
                model = load_model(model_path)
                if model:
                    prediction = get_prediction(model, processed_image)
                    if prediction is not None:
                        # Display prediction with progress bar
                        st.write("Pneumonia Probability:")
                        st.progress(prediction)
                        st.write(f"Probability: {prediction * 100:.1f}%")
                        if prediction > 0.5:
                            st.warning("High probability of pneumonia detected. Consult a medical professional.")
                        else:
                            st.success("Low probability of pneumonia detected.")
            else:
                st.error(f"Model file not found at {model_path}. Please ensure chest_xray_model.h5 is in the 'model' directory.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

# Add model statistics
st.sidebar.header("Model Statistics")
st.sidebar.write("Training samples: 5216")
st.sidebar.write("Validation samples: 856")
st.sidebar.write("Test samples: 624")
st.sidebar.header("Model Performance")
st.sidebar.write("Validation Accuracy: To be updated")
st.sidebar.write("Validation Precision: To be updated")
st.sidebar.write("Validation Recall: To be updated")

# Add information about the model
st.sidebar.header("About")
st.sidebar.info("""
This application uses a deep learning model trained on chest X-ray images to detect pneumonia.
The model was trained on a dataset of chest X-ray images with expert annotations.
""")

# Add usage instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a chest X-ray image using the file uploader
2. The system will automatically analyze the image
3. Results show the probability score for pneumonia
4. Higher percentages indicate higher likelihood of pneumonia
5. Always consult a medical professional for proper diagnosis
""")