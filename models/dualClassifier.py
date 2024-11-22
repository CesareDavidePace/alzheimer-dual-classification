from torchvision import models
from .baseClassifier import BaseClassifier
import torch
import torch.nn as nn

class DualInputClassifier(BaseClassifier):
    def __init__(
        self,
        rgb_backbone: str = "resnet18",
        single_channel_backbone: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
        learning_rate: float = 1e-3
    ):
        super().__init__(num_classes, learning_rate)

        # RGB Backbone
        self.rgb_model = self._load_backbone(rgb_backbone, pretrained, num_classes=None)

        # Single-Channel Backbone
        self.single_channel_model = self._load_backbone(single_channel_backbone, pretrained, num_classes=None, in_channels=1)

        # Combined feature size
        combined_features = self.rgb_model.fc.in_features + self.single_channel_model.fc.in_features
        self.rgb_model.fc = nn.Identity()
        self.single_channel_model.fc = nn.Identity()

        # Classifier on top of concatenated features
        self.classifier = nn.Linear(combined_features, num_classes)

    def _load_backbone(self, backbone_name, pretrained, num_classes, in_channels=3):
        if backbone_name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone_name == "resnet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        # add convnexttiny
        elif backbone_name == "convnexttiny":
            model = models.convnext_tiny(weights=models.ConvNeXTTiny_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if in_channels != 3:
            # Adjust first convolutional layer for custom input channels
            model.conv1 = nn.Conv2d(
                in_channels, model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False
            )
        return model

    def forward(self, rgb_img, single_channel_img):
        # Extract features from RGB and single-channel images
        rgb_features = self.rgb_model(rgb_img)
        single_channel_features = self.single_channel_model(single_channel_img)

        # Concatenate features
        combined_features = torch.cat((rgb_features, single_channel_features), dim=1)

        # Classify
        logits = self.classifier(combined_features)
        return logits

    def _step(self, batch, step_type):
        rgb_img = batch["rgb"]
        single_channel_img = batch["single_channel"]
        labels = batch["label"]

        outputs = self(rgb_img, single_channel_img)
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

    def training_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        single_channel_img = batch["single_channel"]
        labels = batch["label"]

        # Forward pass
        outputs = self(rgb_img, single_channel_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)  # Convert logits to class predictions

        # Update metrics
        self.train_accuracy.update(preds, labels)
        self.train_sensitivity.update(preds, labels)
        self.train_specificity.update(preds, labels)
        self.train_mcc.update(preds, labels)
        self.train_f1.update(preds, labels)

        # Log metrics
        self.log('train_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_sensitivity', self.train_sensitivity, on_step=False, on_epoch=True)
        self.log('train_specificity', self.train_specificity, on_step=False, on_epoch=True)
        self.log('train_mcc', self.train_mcc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        single_channel_img = batch["single_channel"]
        labels = batch["label"]

        # Forward pass
        outputs = self(rgb_img, single_channel_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        self.val_accuracy.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_specificity.update(preds, labels)
        self.val_mcc.update(preds, labels)
        self.val_f1.update(preds, labels)

        # Log metrics
        self.log('val_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_sensitivity', self.val_sensitivity, on_step=False, on_epoch=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True)
        self.log('val_mcc', self.val_mcc, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        single_channel_img = batch["single_channel"]
        labels = batch["label"]

        # Forward pass
        outputs = self(rgb_img, single_channel_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        self.test_accuracy.update(preds, labels)
        self.test_sensitivity.update(preds, labels)
        self.test_specificity.update(preds, labels)
        self.test_mcc.update(preds, labels)
        self.test_f1.update(preds, labels)

        # Log metrics
        self.log('test_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_sensitivity', self.test_sensitivity, on_step=False, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True)
        self.log('test_mcc', self.test_mcc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)

        metrics = {
            "accuracy": self.test_accuracy.compute(),
            "sensitivity": self.test_sensitivity.compute(),
            "specificity": self.test_specificity.compute(),
            "mcc": self.test_mcc.compute(),
            "f1": self.test_f1.compute()
        }

        return metrics, loss