import torchvision.transforms as transforms
from transformers import AutoImageProcessor

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Preprocess images to match EfficientNet's input size and normalization
data_transforms_efficientNet = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Initialize the processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

# Preprocess images to match DINOv2 input size and normalization
data_transforms_DINOV2 = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: processor(images=x, return_tensors='pt')['pixel_values'].squeeze(0))  # Apply processor
    ]
)

# Initialize the processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

# Preprocess images to match DINOv2L input size and normalization
data_transforms_DINOV2L = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: processor(images=x, return_tensors='pt')['pixel_values'].squeeze(0))  # Apply processor
    ]
)

# Initialize the processor
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant-imagenet1k-1-layer")

# Preprocess images to match DINOv2XL input size and normalization
data_transforms_DINOV2XL = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: processor(images=x, return_tensors='pt')['pixel_values'].squeeze(0))  # Apply processor
    ]
)

# Initialize the processor
processor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224")

# Preprocess images to match DINOv2XL input size and normalization
data_transforms_DeiT = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: processor(images=x, return_tensors='pt')['pixel_values'].squeeze(0))  # Apply processor
    ]
)