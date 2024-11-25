import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoImageProcessor, AutoModel

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class BetterNet(nn.Module):
    def __init__(self):
        super(BetterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, nclasses)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
class EfficientNetB7(nn.Module):
    def __init__(self):
        super(EfficientNetB7, self).__init__()
        self.model = models.efficientnet_b7(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, nclasses)

    def forward(self, x):
        return self.model(x)

class DINOv2Model(nn.Module):
    def __init__(self):
        super(DINOv2Model, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, nclasses)

    def forward(self, x):
        outputs = self.model(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits
    
class DINOv2LModel(nn.Module):
    def __init__(self):
        super(DINOv2LModel, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, nclasses)

    def forward(self, x):
        outputs = self.model(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

class DINOv2XLModel(nn.Module):
    def __init__(self):
        super(DINOv2XLModel, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-giant-imagenet1k-1-layer")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, nclasses)

    def forward(self, x):
        outputs = self.model(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits
    
class DeiTModel(nn.Module):
    def __init__(self):
        super(DeiTModel, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/deit-base-patch16-224")
        for param in self.model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.model.config.hidden_size, nclasses)

    def forward(self, x):
        outputs = self.model(x)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits