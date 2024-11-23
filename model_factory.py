"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_efficientNet, data_transforms_DINOV2
from model import Net
from model import BetterNet, DINOv2Model, EfficientNetB7


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "better_cnn":
            return BetterNet()
        if self.model_name == "efficient_net":
            return EfficientNetB7()
        if self.model_name == "dinov2":
            return DINOv2Model()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "better_cnn":
            return data_transforms
        if self.model_name == "efficient_net":
            return data_transforms_efficientNet
        if self.model_name == "dinov2":
            return data_transforms_DINOV2
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
