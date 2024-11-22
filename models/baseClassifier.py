import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    Specificity,
    MatthewsCorrCoef
)
from torchvision import models

class BaseClassifier(pl.LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 1e-3):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Additional metrics
        self.train_sensitivity = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_sensitivity = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.test_sensitivity = Recall(task="multiclass", num_classes=num_classes, average="macro")

        self.train_specificity = Specificity(task="multiclass", num_classes=num_classes, average="macro")
        self.val_specificity = Specificity(task="multiclass", num_classes=num_classes, average="macro")
        self.test_specificity = Specificity(task="multiclass", num_classes=num_classes, average="macro")

        self.train_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.val_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.test_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        labels = batch["label"]

        outputs = self(rgb_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)  # Convert logits to class predictions

        # Update metrics
        self.train_accuracy.update(preds, labels)
        self.train_sensitivity.update(preds, labels)
        self.train_specificity.update(preds, labels)
        self.train_mcc.update(preds, labels)
        self.train_f1.update(preds, labels)

        # Log only the loss for now
        self.log('train_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_sensitivity', self.train_sensitivity, on_step=False, on_epoch=True)
        self.log('train_specificity', self.train_specificity, on_step=False, on_epoch=True)
        self.log('train_mcc', self.train_mcc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        labels = batch["label"]

        outputs = self(rgb_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        self.val_accuracy.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_specificity.update(preds, labels)
        self.val_mcc.update(preds, labels)
        self.val_f1.update(preds, labels)

        # Log only the loss for now
        self.log('val_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_sensitivity', self.val_sensitivity, on_step=False, on_epoch=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True)
        self.log('val_mcc', self.val_mcc, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        rgb_img = batch["rgb"]
        labels = batch["label"]

        outputs = self(rgb_img)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        self.test_accuracy.update(preds, labels)
        self.test_sensitivity.update(preds, labels)
        self.test_specificity.update(preds, labels)
        self.test_mcc.update(preds, labels)
        self.test_f1.update(preds, labels)

        # Log only the loss for now
        self.log('test_loss', loss, batch_size=labels.size(0), on_step=False, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test_sensitivity', self.test_sensitivity, on_step=False, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True)
        self.log('test_mcc', self.test_mcc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)

        metrics = {
            "accuracy": self.test_accuracy,
            "sensitivity": self.test_sensitivity,
            "specificity": self.test_specificity,
            "mcc": self.test_mcc,
            "f1": self.test_f1
        }
        return metrics, loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate) #TODO: add wd paramters
