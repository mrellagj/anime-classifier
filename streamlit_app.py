import streamlit as st
import mlflow
import torchvision
import numpy as np
from PIL import Image
import json


# Set page config
st.set_page_config(
    page_title="Anime Character Classifier",
    page_icon="🎭",
    layout="wide"
)

# Load class mapping
try:
    with open('idx_to_class.json', 'r') as f:
        idx_to_class = json.load(f)
except FileNotFoundError:
    st.error("Class mapping file not found. Please run training first.")
    st.stop()

@st.cache_resource
def load_model():
    import src.models as models
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = models.EfficientNetV2Model(len(idx_to_class))
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(f'models/{type(model).__name__}.pt', map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model weights not found. Please run training first.")
        st.stop()

def preprocess_image(image: Image.Image):
    # Step 1: Resize
    image = image.resize((224, 224))

    # Step 2: Convert to NumPy and scale to [0, 1]
    image_np = np.array(image).astype(np.float32) / 255.0  # Shape: (H, W, C)

    # Step 3: Normalize
    mean = np.array([0.69088995, 0.6272622, 0.64698607])  # Shape: (3,)
    std = np.array([0.28675508, 0.28253487, 0.26916876])   # Shape: (3,)

    # Apply normalization (broadcasting over H, W)
    image_np = (image_np - mean) / std

    # Step 4: Move channel dimension to front (if your model expects CHW)
    image_np = np.transpose(image_np, (2, 0, 1))  # Shape: (C, H, W)

    return image_np

def softmax(x, dim=None):
    """Compute softmax values for each sets of scores in x."""
    # return np.exp(x) / np.sum(np.exp(x), axis=0)
    return x.softmax(dim)

# Main app
st.title("Anime Character Classifier")
st.write("Upload an image to classify the anime character!")

# Load model
# model = mlflow.pyfunc.load_model("models:/ResNet18Model/2")
import torch
torch.classes.__path__ = []
model, device = load_model()
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.69088995, 0.6272622, 0.64698607],
                                     std=[0.28675508, 0.28253487, 0.26916876])
    
])

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with col2, st.spinner("Classifying..."):
        input_tensor = transforms(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_k = torch.topk(probabilities, min(3, len(idx_to_class)))
            
        st.write("### Top 3 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_k.values[0], top_k.indices[0])):
            st.write(f"{i+1}. {idx_to_class[str(idx.item())]}: {prob.item()*100:.2f}%")

st.sidebar.markdown(f"""
### About
This app uses a {type(model).__name__} model trained on anime character images to classify uploaded pictures.

### Instructions
1. Upload an image using the file uploader
2. Wait for the model to process the image
3. View the top 3 predictions with confidence scores
""")