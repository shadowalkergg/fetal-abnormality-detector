import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Ultrasound Anomaly Detection", page_icon="🩺")

img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_extractor():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Identity()
    return m.eval()


@st.cache_resource
def load_models():
    return joblib.load("models.joblib")


extractor = load_extractor()
bundle = load_models()
clfs, categories, accs = bundle["models"], bundle["categories"], bundle["accuracies"]

st.title("🩺 Ultrasound Anomaly Detection")
st.caption("Classifies breast ultrasound images as benign, malignant or normal.")

file = st.file_uploader("Upload an ultrasound image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    with torch.no_grad():
        feat = extractor(img_transform(img).unsqueeze(0)).numpy()

    rows = [
        {"Model": name,
         "Prediction": categories[int(clf.predict(feat)[0])].upper(),
         "Validation accuracy": f"{accs[name]:.2f}%"}
        for name, clf in clfs.items()
    ]
    st.subheader("Predictions")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.warning("For educational purposes only. Not a medical diagnosis.")
