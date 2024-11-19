import streamlit as st
from PIL import Image
from prediction import load_model, predict_flower

# Streamlit app setup
st.title("Flower Identification App 🌼")
st.write("Upload an image of a flower to identify it.")

# Load the model components
with st.spinner("Loading model..."):
    model, preprocessor, id_to_label = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert uploaded file to a format compatible with PIL
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        with st.spinner("Predicting flower type..."):
            predicted_label, confidence = predict_flower(image, model, preprocessor, id_to_label)

        # Display prediction results
        if predicted_label:
            st.success(f"Predicted Flower: {predicted_label} with {confidence:.2f}% confidence.")
        else:
            st.warning("The flower cannot be confidently recognized. Please try another image.")

    except Image.UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
