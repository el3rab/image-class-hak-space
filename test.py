import streamlit as st
import torch
from torchvision import models , transforms
import requests

model = models.resnet18(pretrained = True)
model.eval()

classes = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

transform = transforms.Compose([ 
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(

    )
])

st.title("ـصنيف الصور")
