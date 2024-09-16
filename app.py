import streamlit as st
import torch
from PIL import Image
import numpy as np
from your_gan_model import GANModel  # Import GAN code
from your_autoencoder_model import AutoencoderModel  # Import Autoencoder code

# Initialize models (GAN and Autoencoder)
gan_model = GANModel()
autoencoder_model = AutoencoderModel()

# Streamlit App
st.title("Synthetic Image Generator: GAN & Autoencoder")

# Image Uploader
uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Convert the image to a format suitable for the model
    img_array = np.array(img)
    img_tensor = torch.Tensor(img_array).permute(2, 0, 1).unsqueeze(0)  # Adjusting for model input shape

    # Model Selection
    model_option = st.selectbox("Select the Model for Synthetic Image Generation", ("GAN", "Autoencoder"))

    if st.button("Generate Synthetic Image"):
        if model_option == "GAN":
            # Assuming GAN model function for synthetic data
            synthetic_image = gan_model.generate_synthetic(img_tensor)
        elif model_option == "Autoencoder":
            # Assuming Autoencoder function for synthetic data
            synthetic_image = autoencoder_model.generate_synthetic(img_tensor)
        
        # Convert back to image format to display
        synthetic_image_np = synthetic_image.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
        synthetic_pil_image = Image.fromarray(synthetic_image_np)
        
        # Display synthetic image
        st.image(synthetic_pil_image, caption="Generated Synthetic Image", use_column_width=True)

        # Download synthetic image
        st.download_button(
            label="Download Synthetic Image",
            data=synthetic_pil_image.tobytes(),
            file_name="synthetic_image.png",
            mime="image/png"
        )
