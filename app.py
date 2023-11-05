import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

# Function to perform image enhancement
def enhance_image(image, enhancement_type):
    if enhancement_type == "Bilateral Filtering":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif enhancement_type == "Unsharp Masking":
        blurred = cv2.GaussianBlur(image, (0, 0), 5)
        return cv2.addWeighted(image, 2.5, blurred, -1.5, 0)
    elif enhancement_type == "Saturation Enhancement":
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_img[..., 1] = hsv_img[..., 1] * 1.5  # Increase saturation
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    elif enhancement_type == "Gamma Correction":
        gamma = 1.5
        gamma_corrected = np.power(image / 255.0, gamma)
        return (gamma_corrected * 255).astype(np.uint8)
    elif enhancement_type == "Edge Enhancement (Laplacian)":
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        edge_enhanced_img = image - laplacian
        edge_enhanced_img = cv2.normalize(edge_enhanced_img, None, 0, 255, cv2.NORM_MINMAX)
        return edge_enhanced_img.astype(np.uint8)


# Streamlit app title
st.title("Image Enhancement App")

# Sidebar with enhancement options
st.sidebar.title("Enhancement Options")
enhancement_type = st.sidebar.selectbox("Select Enhancement Type", ["Bilateral Filtering", "Unsharp Masking", "Saturation Enhancement", "Gamma Correction", "Edge Enhancement (Laplacian)"])

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_image)

    # Create columns for original and enhanced images
    col1, col2 = st.columns(2)

    # Display the original image in the left column
    col1.header("Original Image")
    col1.image(image, use_column_width=True)

    # Perform image enhancement
    enhanced_image = enhance_image(np.array(image), enhancement_type)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

    # Display the enhanced image in the right column with empty lines for vertical space
    col2.header(f" Enhanced Image")  # Title
    col2.image(enhanced_image, use_column_width=True, channels="GRAY" if "Edge Enhancement" in enhancement_type else "BGR")

    # Save enhanced image as a temporary PNG file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, enhanced_image)

    # Download enhanced image
    st.markdown("### Download Enhanced Image")
    st.write("Click the link below to download the enhanced image.")
    st.download_button("Download Enhanced Image", temp_filename, f"{enhancement_type}_enhanced_image.png")

    # Clean up temporary file after download
    os.remove(temp_filename)