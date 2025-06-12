import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import json

torch.classes.__path__ = []  

# Set page config
st.set_page_config(
    page_title="Anime Character Classifier",
    page_icon="🎭",
    layout="wide"
)

# Load class mapping
try:
    with open('class_mapping.json', 'r') as f:
        idx_to_class = json.load(f)
except FileNotFoundError:
    st.error("Class mapping file not found. Please run training first.")
    st.stop()

@st.cache_resource
def load_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(idx_to_class))
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model weights not found. Please run training first.")
        st.stop()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Main app
st.title("Anime Character Classifier")
st.write("Upload an image to classify the anime character!")

# Load model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with col2, st.spinner("Classifying..."):
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_k = torch.topk(probabilities, min(3, len(idx_to_class)))
            
        st.write("\r### Top 3 Predictions:")
        for i, (prob, idx) in enumerate(zip(top_k.values[0], top_k.indices[0])):
            st.write(f"{i+1}. {idx_to_class[str(idx.item())]}: {prob.item()*100:.2f}%")

st.sidebar.markdown("""
### About
This app uses a ResNet18 model trained on anime character images to classify uploaded pictures.

### Instructions
1. Upload an image using the file uploader
2. Wait for the model to process the image
3. View the top 3 predictions with confidence scores
""")