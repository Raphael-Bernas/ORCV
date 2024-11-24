from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import requests
from data import data_transforms_DINOV2

# Load a random image from data_sketches/train_images
image = Image.open("data_sketches/train_images/n01484850/img_0j6t0.jpeg")
image = data_transforms_DINOV2(image)
model = AutoModel.from_pretrained('facebook/dinov2-base')
model = model.to("cuda")
model.classifier = nn.Linear(model.config.hidden_size, 500)

inputs = image.to("cuda")
outputs = model(inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)