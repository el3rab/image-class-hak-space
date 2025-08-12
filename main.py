import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests

st.set_page_config("Amr El3RAB")

model = models.resnet18(pretrained=True)
model.eval()

# تحميل أسماء الفئات
classes = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

st.title("images classification -  تصنيف الصور")

uploaded_file = st.file_uploader("Upload Image - ارفع الصورة", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
    _, idx = output.max(1)
    st.write(f"التصنيف: **{classes[idx]}**")
