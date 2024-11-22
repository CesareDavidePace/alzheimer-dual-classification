import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torchvision
from models.custom_model import CustomImageClassifier, CustomMNISTClassifier
from data_module import ImageDataModule, MNISTDataModule
from utils.random_seed import set_random_seed
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


@hydra.main(config_path="config", config_name="config", version_base = "1.1")
def main(cfg: DictConfig):

    # set random seed for reproducibility
    set_random_seed(cfg.seed)

    # Set the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # pytorch version
    print(f"PyTorch version: {torch.__version__}")
    # pytorch lightning version
    print(f"PyTorch Lightning version: {pl.__version__}")
    # torchvision version
    print(f"TorchVision version: {torchvision.__version__}\n")

    # Print available GPUs
    if torch.cuda.is_available():
        print(f"GPU: {cfg.training.gpus}")  
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("No GPU available!")


    # Initialize the data module
    data_module = MNISTDataModule(data_dir=cfg.data.data_dir, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # Initialize the model using parameters from model.yaml
    model = CustomMNISTClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.model.learning_rate,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=cfg.training.monitor_metric,
        mode = cfg.training.monitor_mode,
        save_top_k=cfg.training.save_top_k,
        filename = cfg.training.checkpoint_filename,
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.early_stopping.monitor_metric,  # Metric to monitor
        patience=cfg.training.early_stopping.patience,       # Number of epochs with no improvement
        mode=cfg.training.early_stopping.mode,               # "min" for loss, "max" for accuracy
        verbose=True,
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu",
        devices=len(cfg.training.gpus.split(",")),
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    model = CustomMNISTClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    # Test the model
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()