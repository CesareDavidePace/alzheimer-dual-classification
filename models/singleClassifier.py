from torchvision import models
from .baseClassifier import BaseClassifier
import torch
from torch import nn


class SingleInputClassifier(BaseClassifier):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 10, pretrained: bool = True, learning_rate: float = 1e-3):
        super().__init__(num_classes, learning_rate)

        # Backbone
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify the final layer for the correct number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, rgb_img):
        return self.model(rgb_img)

    def _step(self, batch, step_type):
        rgb_img = batch["rgb"]
        labels = batch["label"]

        outputs = self(rgb_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        accuracy = getattr(self, f"{step_type}_accuracy")
        sensitivity = getattr(self, f"{step_type}_sensitivity")
        specificity = getattr(self, f"{step_type}_specificity")
        mcc = getattr(self, f"{step_type}_mcc")
        f1 = getattr(self, f"{step_type}_f1")

        accuracy.update(preds, labels)
        sensitivity.update(preds, labels)
        specificity.update(preds, labels)
        mcc.update(preds, labels)
        f1.update(preds, labels)

        self.log(f"{step_type}_loss", loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log(f"{step_type}_accuracy", accuracy, on_step=False, on_epoch=True)
        self.log(f"{step_type}_sensitivity", sensitivity, on_step=False, on_epoch=True)
        self.log(f"{step_type}_specificity", specificity, on_step=False, on_epoch=True)
        self.log(f"{step_type}_mcc", mcc, on_step=False, on_epoch=True)
        self.log(f"{step_type}_f1", f1, on_step=False, on_epoch=True)

        return loss